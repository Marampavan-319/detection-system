# Step 2 — Dataset Preparation

This directory contains the training/validation/test dataset for the ElectroDiagnose computer-vision prototype.

## Prototype scope

We are intentionally starting with **12 visible-defect classes** across four device families.

### Laptop

1. `laptop_hinge_damage`
2. `laptop_screen_crack`
3. `laptop_port_damage`

### Smartphone

4. `smartphone_screen_crack`
5. `smartphone_camera_damage`
6. `smartphone_charging_port_damage`

### PCB / Circuit Board

7. `pcb_burn_mark`
8. `pcb_corrosion`
9. `pcb_damaged_connector`

### Router

10. `router_lan_port_damage`
11. `router_antenna_damage`
12. `router_casing_damage`

## Why only 12 classes?

The final product is intended to be device-agnostic, but a 24-hour hackathon prototype needs a measurable scope. These classes are visually identifiable and suitable for a first YOLO demonstration.

The system will **not** claim that every internal electrical failure can be diagnosed from an image. Internal/electrical faults will be handled as possible diagnoses using combined evidence in later steps.

## Target dataset size

For the first prototype:

- Minimum: 20 images/class
- Preferred: 30–50 images/class
- Target total: 360–600 labeled images
- Validation: 15–20%
- Test: 10–15%

If time is limited, prioritize high-quality images and balanced classes over a large dataset.

## YOLO directory structure

The local dataset should eventually look like:

```
datasets/
└── electronics/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
```

Each image has a matching YOLO label file.

Example:

```
images/train/laptop_hinge_001.jpg
labels/train/laptop_hinge_001.txt
```

## YOLO label format

Each object annotation is one line:

```
class_id x_center y_center width height
```

All coordinates are normalized from 0 to 1.

Example:

```
0 0.42 0.58 0.18 0.22
```

## Data quality rules

- Prefer clear, real photographs.
- Include different lighting conditions.
- Include different viewing angles.
- Include different device models/brands where possible.
- Avoid duplicate images.
- Avoid watermarked images when possible.
- Do not put the same physical device/photo in both train and test.
- For defect images, the bounding box should cover the visible abnormal region.
- Keep the original image resolution when collecting; resizing will happen during training.

## Important distinction

A YOLO model will learn **visible patterns**. It should not be presented as proof of an internal electrical failure.

Later, the multimodal reasoning layer will combine:

```
CV findings
+ OCR/error screenshot
+ optional symptoms
+ device/component context
= diagnostic assessment
```

## Dataset sources

For the hackathon, use legally reusable/publicly available images or images you have permission to use. Record the source and license in `datasets/source_log.csv`.

## Next action

Collect and annotate the first batch of images, then train a small YOLO model before expanding the dataset.
