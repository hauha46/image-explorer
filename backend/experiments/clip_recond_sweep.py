"""
Sweep driver for the CLIP text re-conditioning study.

Iterates a grid of ``(input image) x (target prompt) x (lambda)`` runs through
SEVA with the patched ``CLIPConditioner``, saves every artifact required by
the plan (views, contact sheet, per-run metrics, master CSV), and is
crash-tolerant: the master CSV is appended atomically after each run, so a
mid-sweep failure never loses finished runs.

Usage
-----

From the repo root (``backend/`` on PYTHONPATH)::

    python -m backend.experiments.clip_recond_sweep \
        --inputs uploads/living_room_1.jpg uploads/living_room_2.jpg \
        --experiment-root backend/experiments/clip_recond \
        --prompt-set default \
        --lambdas 0.0 0.05 0.1 0.2 0.3 \
        --num-views 10 \
        --skip-consistency   # drop the DUSt3R split reconstruction, ~5x faster

Resume
------

Pass ``--skip-existing`` to skip any ``(input, prompt, lambda)`` cell whose
``run_id`` already appears in ``summary/all_runs.csv``.  That lets you
re-run the script after a crash and finish only the missing cells.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# Make sibling modules importable when launched as ``python -m ...`` or as a
# plain script.
_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent
for p in (_BACKEND_DIR, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments import artifact_layout as al
from experiments import metrics as mx
from experiments.prompts import (
    DEFAULT_LAMBDAS,
    DEFAULT_SWEEP_PROMPTS,
    NEUTRAL_PROMPT,
    full_prompt_list,
)

logger = logging.getLogger("clip_recond_sweep")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="Interior-room input images (paths).")
    p.add_argument(
        "--input-labels",
        nargs="+",
        default=None,
        help="Short labels for each input (defaults to stem of file).",
    )
    p.add_argument(
        "--experiment-root",
        default=str(_THIS_DIR / "clip_recond"),
        help="Where to write all sweep artifacts.",
    )
    p.add_argument(
        "--prompt-set",
        choices=["default", "full"],
        default="default",
        help="`default` = 6 curated prompts; `full` = every prompt in prompts.py.",
    )
    p.add_argument(
        "--extra-prompt",
        nargs=2,
        action="append",
        metavar=("LABEL", "PROMPT"),
        default=[],
        help="Add a one-off prompt (can repeat).",
    )
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDAS,
        help=f"Lambda values to sweep (default: {DEFAULT_LAMBDAS}).",
    )
    p.add_argument("--num-views", type=int, default=10)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--device", default="cuda")
    p.add_argument("--neutral-prompt", default=NEUTRAL_PROMPT)
    p.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="SEVA diffusion steps per sample (default 50). "
             "Use 25 for ~2x speedup with minimal quality loss.",
    )
    p.add_argument(
        "--dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="SEVA backbone / autocast dtype. bf16 is ~2-3x faster on "
             "Ampere/Ada/Blackwell and avoids the fp16-only MATH fallback.",
    )
    p.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also run prompt=None (original SEVA) once per input for baseline metrics.",
    )
    p.add_argument(
        "--skip-consistency",
        action="store_true",
        help="Skip the DUSt3R-based self-consistency metric (~5x faster).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cells whose run already appears in the master CSV.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many runs (useful for smoke tests).",
    )
    p.add_argument("--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_synthesizer_and_dust3r(
    device: str,
    neutral_prompt: str,
    num_steps: int = 50,
    dtype: str = "fp32",
):
    """Lazy-import the heavy deps so this module can be inspected without them."""
    from novel_view_synthesis.seva_synthesizer import SevaSynthesizer
    from reconstruction.dust3r_reconstructor import Dust3rReconstructor

    logger.info("Loading SEVA (num_steps=%d, dtype=%s) ...", num_steps, dtype)
    synth = SevaSynthesizer(
        neutral_prompt=neutral_prompt,
        clip_lambda=0.0,
        num_steps=num_steps,
        dtype=dtype,
    )
    synth.load_model(device=device)

    logger.info("Loading DUSt3R ...")
    recon = Dust3rReconstructor(device=device)
    return synth, recon


def _peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 1e9
    except Exception:
        pass
    return 0.0


def _reset_peak_vram() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _existing_run_ids(experiment_root: str | Path) -> set[str]:
    csv_path = al.master_csv_path(experiment_root)
    if not csv_path.exists():
        return set()
    out: set[str] = set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("run_id")
            if rid:
                out.add(rid)
    return out


def _cell_signature(input_label: str, prompt_label: str, lambda_: float) -> str:
    """Stable signature used for --skip-existing matching."""
    lam = f"{lambda_:.3f}".replace(".", "p")
    return f"{input_label}__{prompt_label}__l{lam}"


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_one(
    synth,
    recon,
    *,
    input_path: str,
    input_label: str,
    prompt_label: str,
    prompt: str | None,
    neutral_prompt: str,
    lambda_: float,
    num_views: int,
    num_steps: int,
    dtype: str,
    seed: int,
    device: str,
    experiment_root: str,
    compute_consistency: bool,
) -> dict:
    run_id = _cell_signature(input_label, prompt_label, lambda_)
    run_dir = al.new_run_dir(experiment_root, run_id)
    views_dir = run_dir / "views"

    config = al.RunConfig(
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        input_image=str(input_path),
        input_label=input_label,
        prompt=prompt,
        prompt_label=prompt_label,
        neutral_prompt=neutral_prompt,
        lambda_=float(lambda_),
        seed=seed,
        cfg=4.0,
        num_views=num_views,
        num_steps=num_steps,
        dtype=dtype,
        model_version="seva-1.1",
        git_sha=al.git_sha(_REPO_ROOT),
    )
    al.write_config(run_dir, config)
    al.write_input_copy(run_dir, input_path)

    logger.info(
        "[run] input=%s prompt=%s lambda=%.3f -> %s",
        input_label, prompt_label, lambda_, run_dir.name,
    )

    _reset_peak_vram()
    t0 = time.time()
    try:
        view_paths = synth.generate_views(
            image_path=str(input_path),
            output_dir=str(run_dir),
            num_views=num_views,
            prompt=prompt,
            neutral_prompt=neutral_prompt,
            clip_lambda=lambda_,
            num_steps=num_steps,
        )
    except Exception:
        logger.error("SEVA generate_views failed for %s:\n%s", run_id, traceback.format_exc())
        runtime_sec = time.time() - t0
        summary = {
            "mean_clip_score": float("nan"),
            "clip_score_view0": float("nan"),
            "mean_lpips_vs_input": float("nan"),
            "self_consistency_chamfer": float("nan"),
            "self_consistency_cpsnr": float("nan"),
            "self_consistency_cssim": float("nan"),
            "error": "sampling_failed",
        }
        al.write_metrics_json(run_dir, summary)
        al.append_master_row(
            experiment_root, config, summary, run_dir, views_dir,
            grid_path=run_dir / "views_grid.png",
            runtime_sec=runtime_sec,
            peak_vram_gb=_peak_vram_gb(),
        )
        return {"run_id": run_id, "status": "failed"}

    runtime_sec = time.time() - t0

    # Keep only actual view files (the SEVA _seva_work staging is already
    # on disk next to views/; leave it for debugging).
    view_files = sorted(Path(views_dir).glob("view_*.png"))
    if not view_files:
        # SEVA may have saved into the generic views_dir under a different
        # naming convention; accept all PNGs as a fallback.
        view_files = sorted(Path(views_dir).glob("*.png"))

    # Contact sheet
    grid_path = run_dir / "views_grid.png"
    try:
        if view_files:
            al.make_contact_sheet(view_files, grid_path)
    except Exception as exc:
        logger.warning("contact sheet failed for %s: %s", run_id, exc)

    # Metrics
    per_view, summary = mx.aggregate(
        generated_paths=[str(p) for p in view_files],
        input_path=str(input_path),
        prompt=prompt or "",
        work_dir=run_dir / "recon_work",
        dust3r_reconstructor=recon if compute_consistency else None,
        device=device,
        compute_consistency=compute_consistency,
    )
    al.write_metrics_json(run_dir, summary)
    al.write_per_view_csv(run_dir, per_view)

    cmd = (
        "python -m backend.experiments.clip_recond_sweep "
        f"--inputs {input_path!r} --input-labels {input_label!r} "
        f"--extra-prompt {prompt_label!r} {prompt!r} "
        f"--lambdas {lambda_} "
        f"--neutral-prompt {neutral_prompt!r} "
        f"--num-views {num_views} --num-steps {num_steps} "
        f"--dtype {dtype} --seed {seed}"
    )
    al.write_run_readme(run_dir, config, cmd)

    peak_vram = _peak_vram_gb()
    al.append_master_row(
        experiment_root, config, summary, run_dir, views_dir, grid_path,
        runtime_sec=runtime_sec, peak_vram_gb=peak_vram,
    )
    logger.info(
        "[done] %s runtime=%.1fs vram=%.1fGB clip=%.3f lpips=%.3f chamfer=%.4f",
        run_id, runtime_sec, peak_vram,
        summary.get("mean_clip_score", float("nan")),
        summary.get("mean_lpips_vs_input", float("nan")),
        summary.get("self_consistency_chamfer", float("nan")),
    )
    return {"run_id": run_id, "status": "ok", "runtime_sec": runtime_sec}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_prompts(args) -> list[tuple[str, str]]:
    if args.prompt_set == "default":
        prompts = list(DEFAULT_SWEEP_PROMPTS)
    else:
        prompts = list(full_prompt_list())
    for label, prompt in args.extra_prompt:
        prompts.append((label, prompt))
    return prompts


def _resolve_labels(args) -> list[str]:
    if args.input_labels:
        if len(args.input_labels) != len(args.inputs):
            raise SystemExit("--input-labels must have one label per --inputs")
        return args.input_labels
    return [Path(p).stem for p in args.inputs]


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    Path(args.experiment_root).mkdir(parents=True, exist_ok=True)
    al.summary_dirs(args.experiment_root)

    input_labels = _resolve_labels(args)
    prompts = _resolve_prompts(args)

    skip_ids = _existing_run_ids(args.experiment_root) if args.skip_existing else set()

    synth, recon = _load_synthesizer_and_dust3r(
        args.device,
        args.neutral_prompt,
        num_steps=args.num_steps,
        dtype=args.dtype,
    )

    total_planned = len(args.inputs) * len(prompts) * len(args.lambdas)
    if args.include_baseline:
        total_planned += len(args.inputs)

    logger.info(
        "Sweep plan: %d inputs x %d prompts x %d lambdas = %d runs (baseline=%s)",
        len(args.inputs), len(prompts), len(args.lambdas), total_planned, args.include_baseline,
    )

    done = 0
    t_total = time.time()
    try:
        for inp, inp_label in zip(args.inputs, input_labels):
            if not Path(inp).exists():
                logger.error("Input not found, skipping: %s", inp)
                continue

            if args.include_baseline:
                sig = _cell_signature(inp_label, "baseline", 0.0)
                if sig in skip_ids:
                    logger.info("Skipping existing baseline: %s", sig)
                else:
                    run_one(
                        synth, recon,
                        input_path=inp, input_label=inp_label,
                        prompt_label="baseline", prompt=None,
                        neutral_prompt=args.neutral_prompt,
                        lambda_=0.0, num_views=args.num_views,
                        num_steps=args.num_steps, dtype=args.dtype,
                        seed=args.seed, device=args.device,
                        experiment_root=args.experiment_root,
                        compute_consistency=not args.skip_consistency,
                    )
                    done += 1
                    if args.limit and done >= args.limit:
                        break

            for p_label, p_text in prompts:
                for lam in args.lambdas:
                    sig = _cell_signature(inp_label, p_label, lam)
                    if sig in skip_ids:
                        logger.info("Skipping existing: %s", sig)
                        continue
                    run_one(
                        synth, recon,
                        input_path=inp, input_label=inp_label,
                        prompt_label=p_label, prompt=p_text,
                        neutral_prompt=args.neutral_prompt,
                        lambda_=float(lam), num_views=args.num_views,
                        num_steps=args.num_steps, dtype=args.dtype,
                        seed=args.seed, device=args.device,
                        experiment_root=args.experiment_root,
                        compute_consistency=not args.skip_consistency,
                    )
                    done += 1
                    if args.limit and done >= args.limit:
                        raise _SweepLimitReached()
    except _SweepLimitReached:
        logger.info("Hit --limit=%d, stopping.", args.limit)

    elapsed = time.time() - t_total
    logger.info("Sweep finished: %d runs in %.1fs (%s)",
                done, elapsed, datetime.now().isoformat())

    # Optional parquet mirror of the master CSV (idempotent).
    try:
        import pandas as pd
        csv_path = al.master_csv_path(args.experiment_root)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            pq_path = al.summary_dirs(args.experiment_root)["root"] / (
                f"sweep_{datetime.now().strftime('%Y%m%d')}.parquet"
            )
            try:
                df.to_parquet(pq_path, index=False)
                logger.info("Wrote parquet snapshot to %s", pq_path)
            except Exception as exc:
                logger.info("Parquet export skipped (%s).", exc)
    except Exception:
        pass

    return 0


class _SweepLimitReached(Exception):
    pass


if __name__ == "__main__":
    raise SystemExit(main())
