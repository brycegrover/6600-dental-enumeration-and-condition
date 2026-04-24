"""
generate_paper_figures.py — EDA and training-curve figures for the paper
========================================================================
Produces the figures the paper references but that aren't already saved to
results/figures/. Reads only existing on-disk artifacts (YOLO label files
and Ultralytics results.csv) — no training or validation is re-run.

Outputs (saved to results/figures/):
  - eda_class_distribution.png     — per-class annotation count (Stage 3 train)
  - eda_bbox_area.png              — normalized bbox area distribution
  - eda_spatial_heatmap.png        — 2D kernel density of box centers
  - eda_objects_per_image.png      — histogram of objects per image
  - training_curves_comparison.png — val mAP@0.5 per epoch for baseline,
                                      stage2, and stage3_curriculum overlaid
  - confusion_matrix_comparison.png — side-by-side normalized confusion
                                       matrices for baseline vs curriculum

Usage:
    python scripts/generate_paper_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE3_LABELS_TRAIN = PROJECT_ROOT / "data" / "processed" / "stage3_disease" / "labels" / "train"
RUNS_DIR = PROJECT_ROOT / "models" / "runs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A467", "#C44E52"]


# yolo label parsing
def parse_yolo_seg_file(path: Path) -> list[tuple[int, np.ndarray]]:
    """Return list of (class_id, polygon (N,2) in normalized coords)."""
    rows: list[tuple[int, np.ndarray]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls = int(parts[0])
        coords = np.array(parts[1:], dtype=float)
        if coords.size % 2 != 0 or coords.size < 6:
            # skip malformed rows (need >= 3 vertices)
            continue
        poly = coords.reshape(-1, 2)
        rows.append((cls, poly))
    return rows


def poly_to_bbox(poly: np.ndarray) -> tuple[float, float, float, float]:
    """(cx, cy, w, h) in normalized coords."""
    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def load_stage3_annotations() -> pd.DataFrame:
    """Load every Stage 3 training annotation into a flat DataFrame."""
    records: list[dict] = []
    label_files = sorted(STAGE3_LABELS_TRAIN.glob("*.txt"))
    for lf in label_files:
        rows = parse_yolo_seg_file(lf)
        for cls, poly in rows:
            cx, cy, w, h = poly_to_bbox(poly)
            records.append({
                "image": lf.stem,
                "class_id": cls,
                "class_name": CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else f"Class {cls}",
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "area": w * h,
            })
    return pd.DataFrame.from_records(records)


# figure 1: class distribution
def fig_class_distribution(df: pd.DataFrame) -> None:
    counts = df["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=CLASS_COLORS)
    for bar, n in zip(bars, counts.values):
        pct = 100.0 * n / total if total else 0.0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{int(n)}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Number of annotations")
    ax.set_title(f"Stage 3 Training Annotations by Class (n={total})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  eda_class_distribution.png — {total} annotations, imbalance ratio {counts.max() / max(counts.min(), 1):.1f}x")


# figure 2: bbox area distribution
def fig_bbox_area(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls_name, color in zip(CLASS_NAMES, CLASS_COLORS):
        sub = df[df["class_name"] == cls_name]["area"]
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=40, range=(0, 0.15), alpha=0.55,
                label=f"{cls_name} (n={len(sub)})", color=color)

    mean_area = df["area"].mean()
    ax.axvline(mean_area, color="k", linestyle="--", linewidth=1,
               label=f"overall mean = {mean_area:.3f}")
    ax.set_xlabel("Normalized bounding-box area")
    ax.set_ylabel("Number of annotations")
    ax.set_title("Bounding-Box Area Distribution by Class")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_bbox_area.png", dpi=150)
    plt.close(fig)
    print(f"  eda_bbox_area.png — mean area {mean_area:.3f} (of full image)")


# figure 3: spatial heatmap
def fig_spatial_heatmap(df: pd.DataFrame) -> None:
    # y inverted so image-top renders on top
    H, xedges, yedges = np.histogram2d(
        df["cx"], df["cy"], bins=40, range=[[0, 1], [0, 1]]
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(
        H.T, origin="upper", extent=[0, 1, 1, 0],
        aspect="auto", cmap="magma",
    )
    ax.set_xlabel("Normalized x (image width)")
    ax.set_ylabel("Normalized y (image height)")
    ax.set_title("Spatial Distribution of Annotation Centers (Stage 3 train)")
    fig.colorbar(im, ax=ax, label="Annotation count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_spatial_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  eda_spatial_heatmap.png — {len(df)} annotation centers")


# figure 4: objects per image
def fig_objects_per_image(df: pd.DataFrame) -> None:
    counts = df.groupby("image").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts, bins=range(0, int(counts.max()) + 2),
            color="#4C72B0", edgecolor="white")
    ax.axvline(counts.mean(), color="red", linestyle="--",
               label=f"mean = {counts.mean():.1f}")
    ax.axvline(counts.median(), color="k", linestyle=":",
               label=f"median = {counts.median():.0f}")
    ax.set_xlabel("Annotations per image")
    ax.set_ylabel("Number of images")
    ax.set_title("Annotations per Image (Stage 3 train)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_objects_per_image.png", dpi=150)
    plt.close(fig)
    print(f"  eda_objects_per_image.png — mean {counts.mean():.1f}, max {counts.max()}")


# figure 5: training curves comparison
def fig_training_curves() -> None:
    """Overlay val mAP@0.5 across epochs for the three comparable runs."""
    runs = {
        "Baseline (single-stage)": RUNS_DIR / "baseline" / "results.csv",
        "Stage 2 (enumeration)":  RUNS_DIR / "stage2" / "results.csv",
        "Stage 3 (curriculum)":    RUNS_DIR / "stage3_curriculum" / "results.csv",
    }

    fig, (ax_box, ax_seg) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#4C72B0", "#DD8452", "#55A467"]
    any_plotted = False

    for (name, path), color in zip(runs.items(), colors):
        if not path.exists():
            print(f"  missing {path} — skipping {name}")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        box_col = "metrics/mAP50(B)"
        seg_col = "metrics/mAP50(M)"
        if box_col in df.columns:
            ax_box.plot(df["epoch"], df[box_col], label=name, color=color, linewidth=2)
            any_plotted = True
        if seg_col in df.columns:
            ax_seg.plot(df["epoch"], df[seg_col], label=name, color=color, linewidth=2)

    for ax, title in [(ax_box, "Box detection"), (ax_seg, "Segmentation")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation mAP@0.5")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if any_plotted:
            ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Training Curves — Curriculum Stages vs. Baseline", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves_comparison.png", dpi=150)
    plt.close(fig)
    print("  training_curves_comparison.png")


# hard-coded test-set numbers from the evaluation runs (paper Tables 2 and 4)
TEST_METRICS = {
    # val numbers are per-metric maxima across epochs (ultralytics convention)
    "Baseline (single-stage)": {
        "box_mAP50":    0.417, "box_mAP50_95":    None,  # 50:95 test numbers not in hand
        "seg_mAP50":    0.400, "seg_mAP50_95":    None,
        "val_box_mAP50": 0.635, "val_seg_mAP50": 0.600,
        "val_box_mAP50_95": 0.415, "val_seg_mAP50_95": 0.374,
    },
    "Stage 3 (Curriculum)": {
        "box_mAP50":    0.394, "box_mAP50_95":    None,
        "seg_mAP50":    0.386, "seg_mAP50_95":    None,
        "val_box_mAP50": 0.558, "val_seg_mAP50": 0.534,
        "val_box_mAP50_95": 0.371, "val_seg_mAP50_95": 0.355,
    },
}

# curriculum per-class AP@0.5 (box), from Table 4
PER_CLASS_AP = {
    "Stage 3 (Curriculum)": {
        "Impacted": 0.942, "Caries": 0.421,
        "Periapical Lesion": 0.193, "Deep Caries": 0.021,
    },
}


# figure 6a: RQ1 test-set bar chart
def fig_rq1_comparison() -> None:
    """Test-set mAP@0.5 for baseline vs curriculum, box and segmentation."""
    metrics = ["Test box mAP@0.5", "Test seg mAP@0.5", "Val box mAP@0.5", "Val seg mAP@0.5"]
    keys    = ["box_mAP50",          "seg_mAP50",          "val_box_mAP50",     "val_seg_mAP50"]
    models  = list(TEST_METRICS.keys())
    colors  = ["#4C72B0", "#55A467"]

    x = np.arange(len(metrics))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [TEST_METRICS[model][k] for k in keys]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=model, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0, 0.72)
    ax.set_title("RQ1 — Curriculum vs Baseline (test and validation)")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rq1_curriculum_vs_baseline.png", dpi=150)
    plt.close(fig)
    print("  rq1_curriculum_vs_baseline.png")


# figure 6b: RQ2 per-stage contribution
def fig_rq2_stage_contribution() -> None:
    """Best validation mAP@0.5 at each stage (each on its own target space),
    plus the baseline for reference. Stage 1 has no surviving results.csv so it
    is shown with a text annotation rather than a numeric bar."""
    def _best(path: Path) -> float | None:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        return float(df["metrics/mAP50(B)"].max()) if "metrics/mAP50(B)" in df.columns else None

    rows = [
        ("Stage 1\n(quadrant,\n4 classes)",      None,                                                   "—"),
        ("Stage 2\n(enumeration,\n8 classes)",    _best(RUNS_DIR / "stage2" / "results.csv"),             "8-class"),
        ("Stage 3\n(curriculum,\n4 classes)",     _best(RUNS_DIR / "stage3_curriculum" / "results.csv"),  "4-class disease"),
        ("Baseline\n(single-stage,\n4 classes)",  _best(RUNS_DIR / "baseline" / "results.csv"),           "4-class disease"),
    ]

    labels = [r[0] for r in rows]
    values = [r[1] if r[1] is not None else 0.0 for r in rows]
    colors_ = ["#CCCCCC", "#DD8452", "#55A467", "#4C72B0"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, values, color=colors_)
    for b, r in zip(bars, rows):
        _, val, target = r
        if val is None:
            ax.text(b.get_x() + b.get_width() / 2, 0.02,
                    "results.csv\nnot archived",
                    ha="center", va="bottom", fontsize=8, style="italic", color="gray")
        else:
            ax.text(b.get_x() + b.get_width() / 2, val + 0.01,
                    f"{val:.3f}\n({target})", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Best validation mAP@0.5 (box)")
    ax.set_ylim(0, 0.75)
    ax.set_title("RQ2 — Best validation mAP@0.5 at each stage (own target space)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rq2_stage_contribution.png", dpi=150)
    plt.close(fig)
    print("  rq2_stage_contribution.png")


# figure 6c: per-class AP (curriculum only)
def fig_per_class_ap() -> None:
    """Curriculum's per-class test AP@0.5. Baseline per-class AP was not
    persisted in the training artifacts (Ultralytics writes aggregate metrics
    only to results.csv), so this panel shows the curriculum run's per-class
    profile on its own."""
    model = "Stage 3 (Curriculum)"
    cls = CLASS_NAMES
    vals = [PER_CLASS_AP[model][c] for c in cls]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(cls, vals, color=CLASS_COLORS)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Test AP@0.5 (box)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-class test AP@0.5 — Curriculum model (DENTEX test set)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.394, color="gray", linestyle="--", linewidth=1,
               label="Overall mAP@0.5 = 0.394")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rq3_per_class_ap.png", dpi=150)
    plt.close(fig)
    print("  rq3_per_class_ap.png (curriculum only; baseline per-class AP not archived)")


# figure 6: confusion-matrix comparison
def fig_confusion_matrices() -> None:
    """Place the Ultralytics-generated normalized confusion matrices for the
    baseline and curriculum runs side-by-side for direct visual comparison."""
    import matplotlib.image as mpimg

    pairs = [
        ("Baseline (single-stage)", RUNS_DIR / "baseline" / "confusion_matrix_normalized.png"),
        ("Curriculum (Stage 3)",    RUNS_DIR / "stage3_curriculum" / "confusion_matrix_normalized.png"),
    ]
    missing = [name for name, path in pairs if not path.exists()]
    if missing:
        print(f"  missing confusion matrices: {missing} — skipping comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (name, path) in zip(axes, pairs):
        ax.imshow(mpimg.imread(str(path)))
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    fig.suptitle("Normalized Confusion Matrices — Baseline vs. Curriculum (validation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_comparison.png", dpi=150)
    plt.close(fig)
    print("  confusion_matrix_comparison.png")


# main
def main() -> None:
    print(f"Loading Stage 3 training annotations from {STAGE3_LABELS_TRAIN} ...")
    df = load_stage3_annotations()
    print(f"  {len(df)} annotations across {df['image'].nunique()} images")
    print("Generating figures:")
    fig_class_distribution(df)
    fig_bbox_area(df)
    fig_spatial_heatmap(df)
    fig_objects_per_image(df)
    fig_training_curves()
    fig_rq1_comparison()
    fig_rq2_stage_contribution()
    fig_per_class_ap()
    fig_confusion_matrices()
    print(f"All figures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
