
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
YAML = PROJECT_ROOT / "data" / "processed" / "yamls" / "stage3_disease.yaml"
CHECKPOINTS = PROJECT_ROOT / "models" / "checkpoints"
RUNS_DIR = CHECKPOINTS / "runs"


# device detection
def get_device(override: str | None = None) -> str:
    if override:
        return override
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# training
def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)

    if not YAML.exists():
        raise FileNotFoundError(
            f"YAML not found: {YAML}\n"
            "Run `python preprocessing.py` first."
        )

    print("Baseline — Single-Stage Diagnosis Detection")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Warm-start: None (ImageNet pretrained)")
    print(f"YAML: {YAML}")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # no curriculum warm-start
    # ImageNet pretrained only
    model = YOLO("yolov8m-seg.pt")

    results = model.train(
        data=str(YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name="baseline",
        exist_ok=True,
        # optimizer (matched to stage 3)
        optimizer="AdamW",
        lr0=0.001,  # higher than stage 3 since no warm-start
        lrf=0.01,
        warmup_epochs=3,
        # regularisation
        weight_decay=0.0005,
        dropout=0.0,
        # augmentation (matched to stage 3)
        augment=True,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.1,
        patience=args.patience,
        save=True,
        plots=True,
        verbose=True,
    )

    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = CHECKPOINTS / "baseline_best.pt"
    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"Baseline checkpoint saved to {best_dst}")
    else:
        print(f"Could not find best.pt at {best_src}")

    print(f"Full run artefacts: {results.save_dir}")
    print("Open models/02_evaluation.ipynb to compare baseline vs. curriculum.")


# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline — Single-Stage Diagnosis Detection")
    p.add_argument("--epochs",   type=int,   default=100,  help="Training epochs")
    p.add_argument("--batch",    type=int,   default=8,    help="Batch size")
    p.add_argument("--imgsz",    type=int,   default=640,  help="Input image size")
    p.add_argument("--patience", type=int,   default=25,   help="Early stopping patience")
    p.add_argument("--device",   type=str,   default=None, help="Device: mps | cuda | cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
