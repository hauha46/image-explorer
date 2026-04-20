"""
Regenerate every figure and table needed for the final report from the master
sweep CSV alone.  No GPU required; idempotent; safe to re-run.

Outputs (all under ``<experiment_root>/summary/``):

    plots/
        clipscore_vs_lambda_<prompt>.png
        lpips_vs_lambda_<prompt>.png
        consistency_vs_lambda_<prompt>.png
        tradeoff_<prompt>.png
    figures/
        qualitative_grid_<input>.png
    report_tables/
        table_prompt_adherence.csv
        table_consistency.csv
        table_best_lambda.csv

Usage::

    python -m backend.experiments.make_report_artifacts \
        --experiment-root backend/experiments/clip_recond
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Non-interactive backend so this runs on headless hosts.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _THIS_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent
for p in (_BACKEND_DIR, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments import artifact_layout as al

logger = logging.getLogger("make_report_artifacts")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_master(experiment_root: str) -> pd.DataFrame:
    csv_path = al.master_csv_path(experiment_root)
    if not csv_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Coerce numeric columns (empty strings -> NaN)
    num_cols = [
        "lambda", "seed", "cfg", "num_views",
        "mean_clip_score", "clip_score_view0", "mean_lpips_vs_input",
        "self_consistency_chamfer", "self_consistency_cpsnr",
        "self_consistency_cssim", "runtime_sec", "peak_vram_gb",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_metric_vs_lambda(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
    title: str,
    invert_y: bool = False,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    for input_label, sub in df.groupby("input_label"):
        sub = sub.sort_values("lambda")
        ax.plot(sub["lambda"], sub[metric], marker="o", label=str(input_label))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if invert_y:
        ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_tradeoff(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(5, 4))
    for input_label, sub in df.groupby("input_label"):
        sub = sub.sort_values("lambda")
        ax.plot(sub["mean_clip_score"], sub["self_consistency_chamfer"],
                marker="o", label=str(input_label))
        # Annotate each point with its lambda
        for _, row in sub.iterrows():
            if not (np.isnan(row["mean_clip_score"]) or np.isnan(row["self_consistency_chamfer"])):
                ax.annotate(f"{row['lambda']:.2f}",
                            (row["mean_clip_score"], row["self_consistency_chamfer"]),
                            fontsize=7, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean CLIP-score (higher = better prompt adherence)")
    ax.set_ylabel("Chamfer distance (lower = better 3D consistency)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_plots(df: pd.DataFrame, out_root: Path) -> list[Path]:
    plots_dir = out_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # One per prompt_label (skip 'baseline' which has no meaningful lambda sweep)
    for p_label, sub in df.groupby("prompt_label"):
        if p_label == "baseline":
            continue
        p1 = plots_dir / f"clipscore_vs_lambda_{p_label}.png"
        _plot_metric_vs_lambda(sub, "mean_clip_score", "mean CLIP-score", p1,
                               f"CLIP-score vs λ — {p_label}")
        written.append(p1)

        p2 = plots_dir / f"lpips_vs_lambda_{p_label}.png"
        _plot_metric_vs_lambda(sub, "mean_lpips_vs_input", "LPIPS (vs input)", p2,
                               f"LPIPS vs λ — {p_label}")
        written.append(p2)

        p3 = plots_dir / f"consistency_vs_lambda_{p_label}.png"
        _plot_metric_vs_lambda(sub, "self_consistency_chamfer", "Chamfer", p3,
                               f"3D self-consistency vs λ — {p_label}")
        written.append(p3)

        p4 = plots_dir / f"tradeoff_{p_label}.png"
        _plot_tradeoff(sub, p4, f"CLIP-score vs consistency tradeoff — {p_label}")
        written.append(p4)

    logger.info("Wrote %d plots to %s", len(written), plots_dir)
    return written


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def make_tables(df: pd.DataFrame, out_root: Path, consistency_tolerance: float = 0.05) -> list[Path]:
    """
    Build the three report tables.

    ``consistency_tolerance`` is the maximum allowed *increase* in Chamfer
    distance (relative to baseline) when picking lambda*.  Defaults to 0.05 in
    normalized units — tune based on actual sweep numbers.
    """
    tables_dir = out_root / "report_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # table_prompt_adherence: per-prompt mean CLIP-score at the smallest and
    # largest lambda actually run.
    if "prompt_label" in df.columns:
        adherence_rows = []
        for p_label, sub in df.groupby("prompt_label"):
            if p_label == "baseline":
                continue
            lam_min = sub["lambda"].min()
            lam_max = sub["lambda"].max()
            row = {
                "prompt_label": p_label,
                "lambda_min": lam_min,
                "clip_score_at_lambda_min": sub[sub["lambda"] == lam_min]["mean_clip_score"].mean(),
                "lambda_max": lam_max,
                "clip_score_at_lambda_max": sub[sub["lambda"] == lam_max]["mean_clip_score"].mean(),
                "delta_clip_score": (
                    sub[sub["lambda"] == lam_max]["mean_clip_score"].mean()
                    - sub[sub["lambda"] == lam_min]["mean_clip_score"].mean()
                ),
            }
            adherence_rows.append(row)
        if adherence_rows:
            df_adh = pd.DataFrame(adherence_rows).sort_values("delta_clip_score", ascending=False)
            p = tables_dir / "table_prompt_adherence.csv"
            df_adh.to_csv(p, index=False, float_format="%.4f")
            written.append(p)

    # table_consistency: Chamfer distance per prompt at smallest vs largest lambda.
    cons_rows = []
    for p_label, sub in df.groupby("prompt_label"):
        if p_label == "baseline":
            continue
        lam_min = sub["lambda"].min()
        lam_max = sub["lambda"].max()
        cons_rows.append({
            "prompt_label": p_label,
            "chamfer_at_lambda_min": sub[sub["lambda"] == lam_min]["self_consistency_chamfer"].mean(),
            "chamfer_at_lambda_max": sub[sub["lambda"] == lam_max]["self_consistency_chamfer"].mean(),
            "delta_chamfer": (
                sub[sub["lambda"] == lam_max]["self_consistency_chamfer"].mean()
                - sub[sub["lambda"] == lam_min]["self_consistency_chamfer"].mean()
            ),
        })
    if cons_rows:
        df_cons = pd.DataFrame(cons_rows)
        p = tables_dir / "table_consistency.csv"
        df_cons.to_csv(p, index=False, float_format="%.4f")
        written.append(p)

    # table_best_lambda: per-prompt, lambda that maximizes CLIP-score subject
    # to chamfer <= baseline_chamfer + consistency_tolerance.
    best_rows = []
    for (input_label, p_label), sub in df.groupby(["input_label", "prompt_label"]):
        if p_label == "baseline":
            continue
        baseline = sub[sub["lambda"] == sub["lambda"].min()]
        if baseline.empty:
            continue
        baseline_cham = baseline["self_consistency_chamfer"].mean()
        threshold = baseline_cham + consistency_tolerance if not np.isnan(baseline_cham) else np.inf
        eligible = sub[sub["self_consistency_chamfer"] <= threshold] if not np.isnan(baseline_cham) else sub
        if eligible.empty:
            continue
        argmax = eligible.loc[eligible["mean_clip_score"].idxmax()]
        best_rows.append({
            "input_label": input_label,
            "prompt_label": p_label,
            "best_lambda": argmax["lambda"],
            "best_clip_score": argmax["mean_clip_score"],
            "best_chamfer": argmax["self_consistency_chamfer"],
            "best_lpips": argmax["mean_lpips_vs_input"],
            "baseline_chamfer": baseline_cham,
            "tolerance": consistency_tolerance,
        })
    if best_rows:
        df_best = pd.DataFrame(best_rows).sort_values(["input_label", "prompt_label"])
        p = tables_dir / "table_best_lambda.csv"
        df_best.to_csv(p, index=False, float_format="%.4f")
        written.append(p)

    logger.info("Wrote %d tables to %s", len(written), tables_dir)
    return written


# ---------------------------------------------------------------------------
# Qualitative grid
# ---------------------------------------------------------------------------

def _load_thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        im = Image.new("RGB", size, (32, 32, 32))
    return im.resize(size, Image.LANCZOS)


def _text_label(text: str, width: int, height: int = 28) -> Image.Image:
    im = Image.new("RGB", (width, height), (255, 255, 255))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    d = ImageDraw.Draw(im)
    d.text((4, 6), text[:120], fill=(0, 0, 0), font=font)
    return im


def make_qualitative_grid(
    df: pd.DataFrame,
    experiment_root: str,
    out_path: Path,
    input_label: str,
    prompts: list[str] | None = None,
    lambdas: list[float] | None = None,
    thumb: tuple[int, int] = (256, 256),
    frame_idx: int | None = None,
) -> Path | None:
    """
    Build a qualitative ``rows = prompts`` x ``cols = lambdas`` image grid
    using the first (or ``frame_idx``-th) saved view of each run for the given
    ``input_label``.
    """
    sub = df[df["input_label"] == input_label]
    if sub.empty:
        logger.warning("qualitative grid: no rows for input_label=%r", input_label)
        return None

    if prompts is None:
        prompts = [p for p in sub["prompt_label"].unique().tolist() if p != "baseline"]
    if lambdas is None:
        lambdas = sorted(sub["lambda"].unique().tolist())

    if not prompts or not lambdas:
        return None

    thumb_w, thumb_h = thumb
    label_h = 28
    row_label_w = 140

    grid_w = row_label_w + thumb_w * len(lambdas)
    grid_h = label_h + thumb_h * len(prompts)
    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    # Column (lambda) labels
    for c, lam in enumerate(lambdas):
        lbl = _text_label(f"λ={lam:.2f}", thumb_w, label_h)
        canvas.paste(lbl, (row_label_w + c * thumb_w, 0))

    for r, p_label in enumerate(prompts):
        # Row (prompt) label
        lbl = _text_label(p_label, row_label_w, thumb_h)
        canvas.paste(lbl, (0, label_h + r * thumb_h))

        for c, lam in enumerate(lambdas):
            cell = sub[(sub["prompt_label"] == p_label) & (sub["lambda"] == lam)]
            if cell.empty:
                im = Image.new("RGB", (thumb_w, thumb_h), (0, 0, 0))
            else:
                views_dir = Path(cell.iloc[0]["views_dir"])
                view_files = sorted(views_dir.glob("view_*.png"))
                if not view_files:
                    view_files = sorted(views_dir.glob("*.png"))
                if not view_files:
                    im = Image.new("RGB", (thumb_w, thumb_h), (64, 0, 0))
                else:
                    idx = 0 if frame_idx is None else max(0, min(frame_idx, len(view_files) - 1))
                    im = _load_thumb(view_files[idx], (thumb_w, thumb_h))
            canvas.paste(im, (row_label_w + c * thumb_w, label_h + r * thumb_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out_path))
    logger.info("Qualitative grid -> %s", out_path)
    return out_path


def make_all_qualitative_grids(df: pd.DataFrame, out_root: Path) -> list[Path]:
    figs_dir = out_root / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for input_label in df["input_label"].unique():
        out_path = figs_dir / f"qualitative_grid_{input_label}.png"
        p = make_qualitative_grid(df, str(out_root.parent), out_path, input_label)
        if p is not None:
            written.append(p)
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-root", default=str(_THIS_DIR / "clip_recond"))
    parser.add_argument(
        "--consistency-tolerance", type=float, default=0.05,
        help="Max allowed increase in Chamfer (vs baseline) when picking best lambda.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    df = load_master(args.experiment_root)
    logger.info("Loaded %d sweep rows from %s",
                len(df), al.master_csv_path(args.experiment_root))

    out_root = al.summary_dirs(args.experiment_root)["root"]

    written: list[Path] = []
    written += make_plots(df, out_root)
    written += make_tables(df, out_root, consistency_tolerance=args.consistency_tolerance)
    written += make_all_qualitative_grids(df, out_root)

    logger.info("Regenerated %d report artifacts under %s", len(written), out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
