# ElectroDiagnose

AI-powered visual and multimodal diagnostic assistant for electronic devices.

## What are we building?

ElectroDiagnose is designed so a user does **not** need to know what is wrong with an electronic device.

The user can provide:

- Device photographs
- Internal/component photographs
- Software or error screenshots
- Optional symptom description

The system will progressively analyze the evidence and produce:

- Device identification
- Suspected component or subsystem
- Visible defect
- Confidence
- Severity
- Possible cause
- Recommended next action

## Prototype Categories

The hackathon prototype demonstrates the approach on:

1. Laptop
2. Smartphone
3. PCB / circuit board
4. Router

The architecture is designed to be device-agnostic, while the prototype is validated on selected categories.

## Planned Technology Stack

- Python
- Streamlit
- OpenCV
- Pillow
- NumPy / Pandas
- Ultralytics YOLO + PyTorch
- Multimodal vision AI
- EasyOCR
- SQLite
- Plotly
- ReportLab

## Repository Structure

```
ElectroDiagnose/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
├── datasets/
├── uploads/
├── outputs/
├── reports/
├── data/
└── src/
    ├── __init__.py
    ├── detection.py
    ├── diagnosis.py
    ├── ocr.py
    ├── severity.py
    └── database.py
```

Runtime directories will be populated as the prototype is implemented.

## Development Roadmap

### Step 1 — Project foundation
Repository, Streamlit starter UI, dependencies, and module scaffolding.

### Step 2 — Dataset preparation
Collect and organize prototype images for laptops, smartphones, PCBs, and routers. The repository now contains a local-only dataset manifest, unified YOLO class list, YOLO configuration, and a preparation script. The pipeline does not clone or depend on another GitHub repository.

### Step 3 — Image processing
Implemented deterministic image preprocessing in `src/preprocessing.py`: EXIF orientation correction, RGB conversion, aspect-ratio-preserving 640x640 letterbox resizing, and float32 normalization to [0, 1]. `scripts/validate_preprocessing.py` validates individual images or directories before inference. The preprocessing stage preserves the original image and keeps normalized data in memory rather than rewriting source images.

### Step 4 — Computer vision
Connect YOLO for device/component/defect detection.

### Step 5 — Component identification
Map detected regions to device components and subsystems.

### Step 6 — Defect localization
Detect and annotate visible abnormalities.

### Step 7 — Multimodal reasoning
Combine visual findings, OCR/error screenshots, and optional symptoms.

### Step 8 — Severity and confidence
Generate structured confidence, severity, and recommendation values.

### Step 9 — Diagnostic result engine
Create a consistent diagnostic response format.

### Step 10 — Streamlit product UI
Connect all analysis modules to the user-facing dashboard.

### Step 11 — Database and report
Store diagnostic history and generate PDF reports.

### Step 12 — Testing and demo
Validate the four prototype scenarios and prepare the hackathon demonstration.

## Local Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Safety

This is an AI-assisted diagnostic prototype. Visual evidence cannot by itself prove an internal electrical fault. Results should be treated as likely assessments and should not replace qualified repair or electrical inspection, especially for mains-powered or high-energy equipment.
