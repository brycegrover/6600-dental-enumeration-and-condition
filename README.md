# Dental Project

Hierarchical dental diagnosis on panoramic X-rays using YOLOv8 and the DENTEX dataset (MICCAI 2023).

## Goal

Train a YOLOv8 model in three curriculum stages (quadrant → tooth → diagnosis) and produce a per-patient treatment priority report.

## Dataset

DENTEX — panoramic X-rays from three clinics, FDI-numbered.

- Tier 1 (quadrant only): 693 images
- Tier 2 (quadrant + enumeration): 634 images
- Tier 3 (full labels + masks): 705 images
- Unlabeled: 1,571 images (not used)

Diagnosis classes: Caries, Deep Caries, Periapical Lesion, Impacted Tooth.

Test split: 250 images.

## Research Questions

RQ1 — Does curriculum training improve performance?

RQ2 — Which curriculum stage contributes most to final model performance?

RQ3 — To what extent do realistic image degradations such as blur, noise, and motion reduce classification performance, and can augmentation with these degradations improve model robustness?

RQ4 — To what extent does region-focused preprocessing at different anatomical scales improve classification performance and robustness compared with using the full panoramic image?

## Methods

Model: `yolov8m-seg` (Ultralytics), trained on MPS or CUDA.

Curriculum:

- Stage 1: quadrant detection on Tier 1 + Tier 3 (1,398 images)
- Stage 2: tooth enumeration on Tier 2 + Tier 3 (1,339 images)
- Stage 3: diagnosis detection + segmentation on Tier 3 (705 images)

Each stage initializes from the previous stage's checkpoint.

Evaluation: mAP@0.5 and mAP@0.5:0.95 on the 250-image test set, per class. Baseline is a single-stage YOLOv8 trained only on Tier 3.

Priority scoring per tooth uses diagnosis class weight, detection confidence, and co-occurring findings. Teeth are ranked per patient.

## Structure

```
Dental_Project/
├── data/raw/dentex/DENTEX/
├── notebooks/
├── src/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── output/
├── models/checkpoints/
├── results/figures/
└── README.md
```

## References

1. Hamamci et al. (2023). *DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays.* arXiv:2305.19112.
2. Hamamci et al. (2023). *Diffusion-based Hierarchical Multi-Label Object Detection to Analyze Panoramic Dental X-rays.* MICCAI 2023.
3. Jocher et al. (2023). *Ultralytics YOLOv8.* https://github.com/ultralytics/ultralytics
