# !/usr/bin/env python3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
YAML_DIR = PROCESSED / "yamls"


def write_yaml(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"  Written: {path}")


def build_yamls():
    YAML_DIR.mkdir(parents=True, exist_ok=True)

    # stage 1
    s1 = PROCESSED / "stage1_quadrant"
    write_yaml(YAML_DIR / "stage1_quadrant.yaml",
f"""# Stage 1 - Quadrant Detection
# Training: 693 (quadrant) + 705 (disease) = 1,398 images

path: {s1.resolve()}
train: images/train
val: images/val

nc: 4
names:
  0: quadrant_1
  1: quadrant_2
  2: quadrant_3
  3: quadrant_4
""")

    # stage 2
    s2 = PROCESSED / "stage2_enumeration"
    write_yaml(YAML_DIR / "stage2_enumeration.yaml",
f"""# Stage 2 - Tooth Enumeration
# Training: 634 (enumeration) + 705 (disease) = 1,339 images
# Init from Stage 1 checkpoint.

path: {s2.resolve()}
train: images/train
val: images/val

nc: 8
names:
  0: tooth_1
  1: tooth_2
  2: tooth_3
  3: tooth_4
  4: tooth_5
  5: tooth_6
  6: tooth_7
  7: tooth_8
""")

    # stage 3
    s3 = PROCESSED / "stage3_disease"
    write_yaml(YAML_DIR / "stage3_disease.yaml",
f"""# Stage 3 - Diagnosis Detection + Segmentation
# Training: 705 images. Init from Stage 2 checkpoint.

path: {s3.resolve()}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: Impacted
  1: Caries
  2: Periapical_Lesion
  3: Deep_Caries
""")

    print(f"All YAMLs written to {YAML_DIR}")


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    build_yamls()
