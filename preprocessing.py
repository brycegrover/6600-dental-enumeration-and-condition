#!/usr/bin/env python3
"""
preprocessing.py
-----------------
Master preprocessing script for the DENTEX dental X-ray project.
Runs all three preprocessing stages in order:

  1. Convert COCO training/validation JSONs -> YOLO segmentation labels
  2. Convert LabelMe test JSONs -> YOLO segmentation labels
  3. Generate YOLOv8 dataset YAML configs

Output lands in data/processed/.

Run from the project root:
    python preprocessing.py
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PROCESSED = PROJECT_ROOT / "data" / "processed"

STEPS = [
    ("01_convert_training_to_yolo.py", "Converting training/validation COCO -> YOLO labels"),
    ("02_convert_test_to_yolo.py",     "Converting test LabelMe -> YOLO labels"),
    ("03_generate_yamls.py",           "Generating dataset YAML configs"),
]


def run_step(script_name, description):
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"  ERROR: {script_path} not found")
        return False
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"  FAILED with exit code {result.returncode}")
        return False
    return True


def verify_output():
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    checks = []
    for stage, expected_splits in [
        ("stage1_quadrant",    ["train", "val"]),
        ("stage2_enumeration", ["train", "val"]),
        ("stage3_disease",     ["train", "val", "test"]),
    ]:
        for split in expected_splits:
            img_dir = PROCESSED / stage / "images" / split
            lbl_dir = PROCESSED / stage / "labels" / split
            n_imgs = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
            n_lbls = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
            match = n_imgs == n_lbls and n_imgs > 0
            checks.append(match)
            print(f"  {stage}/{split}: {n_imgs} images, {n_lbls} labels [{'OK' if match else 'MISMATCH'}]")
    yaml_dir = PROCESSED / "yamls"
    for name in ["stage1_quadrant.yaml", "stage2_enumeration.yaml", "stage3_disease.yaml"]:
        exists = (yaml_dir / name).exists()
        checks.append(exists)
        print(f"  yamls/{name}: {'OK' if exists else 'MISSING'}")
    return all(checks)


def main():
    print("=" * 60)
    print("DENTEX Preprocessing Pipeline")
    print("=" * 60)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {PROCESSED}")

    # Pre-flight: check raw data exists
    raw = PROJECT_ROOT / "data" / "raw"
    required = [
        raw / "training" / "training_data" / "quadrant" / "train_quadrant.json",
        raw / "training" / "training_data" / "quadrant_enumeration" / "train_quadrant_enumeration.json",
        raw / "training" / "training_data" / "quadrant-enumeration-disease" / "train_quadrant_enumeration_disease.json",
        raw / "validation_triple.json",
        raw / "test" / "disease" / "label",
        raw / "test" / "disease" / "input",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("\nERROR: Missing required raw data:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)
    print("Pre-flight check: all raw data present")

    # Clean previous output so re-runs work cleanly
    if PROCESSED.exists():
        print(f"Cleaning previous output: {PROCESSED}")
        shutil.rmtree(PROCESSED)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, (script, desc) in enumerate(STEPS, 1):
        print(f"\n{'─' * 60}")
        print(f"Step {i}/3: {desc}")
        print(f"{'─' * 60}")
        ok = run_step(script, desc)
        if not ok:
            print(f"\nPipeline failed at step {i}. Fix the error and re-run.")
            sys.exit(1)

    elapsed = time.time() - t0
    all_ok = verify_output()

    print(f"\n{'=' * 60}")
    if all_ok:
        print(f"ALL DONE  ({elapsed:.1f}s)")
        print(f"Processed data: {PROCESSED}")
    else:
        print("DONE WITH WARNINGS — check output above")
    print("=" * 60)


if __name__ == "__main__":
    main()
