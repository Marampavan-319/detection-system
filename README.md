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
Implemented a lazy-loading Ultralytics YOLO inference wrapper in src/detection.py with configurable confidence and IoU thresholds and a stable detection schema. scripts/validate_detection.py provides a reproducible smoke test, while configs/yolo_inference.yaml records default inference settings.

### Step 5 — Component identification
Implemented `src/components.py` to map detector classes into device components and subsystems for PCB, laptop, smartphone, and router categories. The mapping layer is explicit and deterministic, preserves detector confidence and bounding boxes, and falls back to unknown component/subsystem when a class is not mapped. `scripts/validate_components.py` validates the mapping independently of model inference.

### Step 6 — Defect localization
Implemented `src/localization.py` to convert component matches into structured visible-defect locations. Each localization preserves the image-space bounding box and confidence while adding the defect center, normalized bounding box, bounding-box area ratio, and a coarse image region such as `middle-center` or `middle-right`. Coordinates are clipped to the image dimensions so downstream modules receive safe, consistent geometry. `scripts/validate_localization.py` provides a deterministic smoke test.

### Step 7 — Multimodal reasoning
Implemented `src/reasoning.py` to combine structured visual/localization findings, OCR or error-screen text, and optional user symptoms into an evidence-based reasoning result. The module produces observations, possible causes, next actions, and explicit limitations without claiming that visible evidence proves an internal electrical fault. `scripts/validate_reasoning.py` provides a deterministic smoke test.

### Step 8 — Severity and confidence
Implemented `src/severity.py` to convert localized visual evidence into deterministic confidence, severity, and priority values with an explainable rationale. The scorer can incorporate the structured reasoning evidence from Step 7 and explicitly distinguishes visible-risk scoring from proof of an internal electrical fault. `scripts/validate_severity.py` covers high, medium, empty-evidence, and router connection-state cases.

### Step 9 — Diagnostic result engine
Implemented `src/diagnosis.py` to combine defect localization, multimodal reasoning, and severity/confidence into one consistent JSON-friendly diagnostic result. The engine reports status, device, findings, confidence, severity, priority, possible causes, next actions, limitations, and whether human review is required. It uses `INSUFFICIENT_EVIDENCE` when no visible defect is localized rather than treating a missing detection as proof that the device is healthy. `scripts/validate_diagnosis.py` validates both a detected PCB short and an empty-evidence case.

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
