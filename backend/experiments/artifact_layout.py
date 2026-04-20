"""
Run-artifact layout for the CLIP text re-conditioning sweep.

Directory layout (relative to ``--experiment-root`` passed on the CLI,
default ``backend/experiments/clip_recond``)::

    <experiment_root>/
        <timestamp>_<run_id>/
            config.json              # full reproducible config
            input.png                # copy of the input image used
            views/
                view_000.png ...     # generated novel views
            views_grid.png           # contact-sheet image for qualitative figures
            metrics.json             # aggregate metrics
            per_view_metrics.csv     # row-per-view metrics
            README.md                # auto-generated reproduction instructions
            recon_work/              # DUSt3R working dir (split_even/, split_odd/, ...)
        summary/
            all_runs.csv             # master table appended after every run
            sweep_<date>.parquet     # same data as parquet (if pyarrow available)
            plots/
            figures/
            report_tables/
"""
from __future__ import annotations

import csv
import json
import logging
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

logger = logging.getLogger(__name__)


# Flat list of columns written to summary/all_runs.csv.  Keep in sync with
# ``RunConfig`` + the metrics keys emitted by ``experiments.metrics.aggregate``.
MASTER_CSV_COLUMNS: list[str] = [
    "run_id",
    "timestamp",
    "input_image",
    "input_label",
    "prompt",
    "prompt_label",
    "neutral_prompt",
    "lambda",
    "seed",
    "cfg",
    "num_views",
    "num_steps",
    "dtype",
    "model_version",
    "git_sha",
    "mean_clip_score",
    "clip_score_view0",
    "mean_lpips_vs_input",
    "self_consistency_chamfer",
    "self_consistency_cpsnr",
    "self_consistency_cssim",
    "runtime_sec",
    "peak_vram_gb",
    "views_dir",
    "grid_path",
    "run_dir",
]


@dataclass
class RunConfig:
    """Full reproducible description of a single sweep run."""
    run_id: str
    timestamp: str
    input_image: str
    input_label: str
    prompt: str | None
    prompt_label: str
    neutral_prompt: str
    lambda_: float
    seed: int = 23
    cfg: float = 4.0
    num_views: int = 10
    num_steps: int = 50
    dtype: str = "fp32"
    model_version: str = "seva-1.1"
    git_sha: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lambda"] = d.pop("lambda_")
        return d


# ---------------------------------------------------------------------------
# Paths / filesystem helpers
# ---------------------------------------------------------------------------

def make_run_id(input_label: str, prompt_label: str, lambda_: float) -> str:
    """Deterministic-ish run id: ``<input>_<prompt>_l<lambda>_<HHMMSS>``."""
    t = datetime.now().strftime("%H%M%S")
    lam = f"{lambda_:.3f}".replace(".", "p")
    safe = lambda s: "".join(c if c.isalnum() or c in "-_" else "_" for c in s)[:32]
    return f"{safe(input_label)}_{safe(prompt_label)}_l{lam}_{t}"


def new_run_dir(experiment_root: str | Path, run_id: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(experiment_root) / f"{ts}_{run_id}"
    (run_dir / "views").mkdir(parents=True, exist_ok=True)
    (run_dir / "recon_work").mkdir(parents=True, exist_ok=True)
    return run_dir


def summary_dirs(experiment_root: str | Path) -> dict[str, Path]:
    root = Path(experiment_root) / "summary"
    paths = {
        "root": root,
        "plots": root / "plots",
        "figures": root / "figures",
        "report_tables": root / "report_tables",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def master_csv_path(experiment_root: str | Path) -> Path:
    root = summary_dirs(experiment_root)["root"]
    return root / "all_runs.csv"


def git_sha(repo_root: str | Path | None = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Config + per-run artifact writers
# ---------------------------------------------------------------------------

def write_config(run_dir: Path, config: RunConfig) -> Path:
    p = run_dir / "config.json"
    p.write_text(json.dumps(config.to_json_dict(), indent=2), encoding="utf-8")
    return p


def write_input_copy(run_dir: Path, input_image: str | Path) -> Path:
    p = run_dir / "input.png"
    try:
        img = Image.open(input_image).convert("RGB")
        img.save(p)
    except Exception:
        shutil.copy2(str(input_image), str(p))
    return p


def make_contact_sheet(
    view_paths: Sequence[str | Path],
    output_path: str | Path,
    cols: int | None = None,
    thumb_w: int = 256,
) -> Path:
    """Write a contact-sheet image combining all ``view_paths``."""
    view_paths = list(view_paths)
    n = len(view_paths)
    if n == 0:
        raise ValueError("make_contact_sheet: no views to composite")
    if cols is None:
        cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    imgs = [Image.open(p).convert("RGB") for p in view_paths]
    # Uniform thumb height preserving aspect ratio of the first image.
    aspect = imgs[0].height / imgs[0].width
    thumb_h = max(1, int(round(thumb_w * aspect)))
    imgs = [im.resize((thumb_w, thumb_h), Image.LANCZOS) for im in imgs]

    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (0, 0, 0))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * thumb_w, r * thumb_h))
    sheet.save(str(output_path))
    return Path(output_path)


def write_metrics_json(run_dir: Path, summary: dict[str, Any]) -> Path:
    p = run_dir / "metrics.json"
    p.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return p


def write_per_view_csv(run_dir: Path, per_view: dict[int, dict]) -> Path:
    p = run_dir / "per_view_metrics.csv"
    if not per_view:
        p.write_text("frame_idx\n", encoding="utf-8")
        return p
    keys = sorted(next(iter(per_view.values())).keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for idx in sorted(per_view):
            w.writerow({k: per_view[idx].get(k, "") for k in keys})
    return p


def write_run_readme(run_dir: Path, config: RunConfig, cmd: str) -> Path:
    lines = [
        f"# Sweep run `{config.run_id}`",
        "",
        f"- **input**: `{config.input_image}`",
        f"- **prompt**: {config.prompt!r}",
        f"- **neutral_prompt**: {config.neutral_prompt!r}",
        f"- **lambda**: {config.lambda_}",
        f"- **seed**: {config.seed}",
        f"- **cfg**: {config.cfg}",
        f"- **num_views**: {config.num_views}",
        f"- **num_steps**: {config.num_steps}",
        f"- **dtype**: {config.dtype}",
        f"- **model_version**: {config.model_version}",
        f"- **git_sha**: `{config.git_sha or '(unknown)'}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        cmd,
        "```",
        "",
        "All artifacts in this folder were produced by",
        "`backend/experiments/clip_recond_sweep.py`.",
    ]
    p = run_dir / "README.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Master CSV append (atomic enough for sweep-level crash safety)
# ---------------------------------------------------------------------------

def append_master_row(
    experiment_root: str | Path,
    config: RunConfig,
    summary: dict[str, Any],
    run_dir: Path,
    views_dir: Path,
    grid_path: Path,
    runtime_sec: float,
    peak_vram_gb: float,
) -> Path:
    csv_path = master_csv_path(experiment_root)
    row = {
        "run_id": config.run_id,
        "timestamp": config.timestamp,
        "input_image": config.input_image,
        "input_label": config.input_label,
        "prompt": config.prompt or "",
        "prompt_label": config.prompt_label,
        "neutral_prompt": config.neutral_prompt,
        "lambda": config.lambda_,
        "seed": config.seed,
        "cfg": config.cfg,
        "num_views": config.num_views,
        "num_steps": config.num_steps,
        "dtype": config.dtype,
        "model_version": config.model_version,
        "git_sha": config.git_sha,
        "mean_clip_score": summary.get("mean_clip_score"),
        "clip_score_view0": summary.get("clip_score_view0"),
        "mean_lpips_vs_input": summary.get("mean_lpips_vs_input"),
        "self_consistency_chamfer": summary.get("self_consistency_chamfer"),
        "self_consistency_cpsnr": summary.get("self_consistency_cpsnr"),
        "self_consistency_cssim": summary.get("self_consistency_cssim"),
        "runtime_sec": runtime_sec,
        "peak_vram_gb": peak_vram_gb,
        "views_dir": str(views_dir),
        "grid_path": str(grid_path),
        "run_dir": str(run_dir),
    }

    # Append atomically: write a .tmp next to the CSV, then os.replace.
    import os, tempfile
    exists = csv_path.exists()

    # Normalize values (NaN -> empty string) for csv readability.
    clean_row = {k: ("" if (isinstance(v, float) and math.isnan(v)) else v) for k, v in row.items()}

    if not exists:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=MASTER_CSV_COLUMNS)
            w.writeheader()
            w.writerow(clean_row)
        return csv_path

    # Append: read existing content then rewrite via temp file.
    existing = csv_path.read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", delete=False, dir=str(csv_path.parent)
    ) as tmp:
        tmp.write(existing)
        if not existing.endswith("\n"):
            tmp.write("\n")
        w = csv.DictWriter(tmp, fieldnames=MASTER_CSV_COLUMNS)
        w.writerow(clean_row)
        tmp_name = tmp.name
    os.replace(tmp_name, csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_default(x: Any):
    try:
        import numpy as np
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"Not JSON serializable: {type(x)!r}")
