"""Streamlit product dashboard for ElectroDiagnose."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from src.components import identify_components
from src.database import list_diagnoses, save_diagnosis
from src.report import generate_pdf_report
from src.diagnosis import diagnose
from src.localization import localize_defects
from src.reasoning import reason_about_evidence
from src.severity import score_severity


DEVICE_OPTIONS = ["Laptop", "Smartphone", "PCB", "Router"]


def _device_key(device: str) -> str:
    return device.strip().lower()


def _demo_detections(device: str, image: Image.Image) -> list[dict[str, Any]]:
    """Return clearly-labelled deterministic demo detections for presentation."""
    width, height = image.size
    category = _device_key(device)

    demo_classes = {
        "pcb": ("short", 0.91, (0.30, 0.30, 0.70, 0.70)),
        "laptop": ("screen", 0.84, (0.20, 0.18, 0.80, 0.65)),
        "smartphone": ("screen", 0.86, (0.24, 0.12, 0.76, 0.88)),
        "router": ("led off", 0.88, (0.62, 0.12, 0.82, 0.32)),
    }
    class_name, confidence, rel_box = demo_classes[category]
    x1, y1, x2, y2 = rel_box
    return [
        {
            "class_id": 0,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": [x1 * width, y1 * height, x2 * width, y2 * height],
        }
    ]


def _run_yolo(device: str, image: Image.Image, model_path: str, confidence: float, iou: float):
    """Run the existing YOLO wrapper without importing Ultralytics at app startup."""
    from src.detection import YOLODetector

    detector = YOLODetector(
        model_path=model_path,
        confidence=confidence,
        iou=iou,
    )
    return [item.to_dict() for item in detector.detect(image)]


def analyze_evidence(
    device: str,
    image: Image.Image,
    detections: list[dict[str, Any]],
    ocr_text: str = "",
    symptoms: str = "",
) -> dict[str, Any]:
    """Connect Sessions 5-9 into one application-level analysis pipeline."""
    width, height = image.size
    matches = identify_components(device, detections)
    locations = localize_defects(matches, width, height)
    reasoning = reason_about_evidence(
        device=device,
        locations=locations,
        ocr_text=ocr_text,
        symptoms=symptoms,
    )
    severity = score_severity(locations, reasoning)
    result = diagnose(
        device=device,
        locations=locations,
        reasoning=reasoning,
        severity=severity,
    )
    return {
        "detections": detections,
        "components": matches,
        "locations": locations,
        "reasoning": reasoning,
        "severity": severity,
        "diagnosis": result,
    }


def draw_findings(image: Image.Image, locations: list[dict[str, Any]]) -> Image.Image:
    """Render localized findings on a copy of the uploaded image."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for index, item in enumerate(locations, start=1):
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox"]]
        label = (
            f"{index}. {item['defect_class']} | "
            f"{item['confidence']:.2f}"
        )
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        text_box = draw.textbbox((x1, max(0, y1 - 22)), label)
        draw.rectangle(text_box, fill="red")
        draw.text((x1 + 4, max(0, y1 - 20)), label, fill="white")
    return canvas


def _show_result(result: dict[str, Any], annotated: Image.Image) -> None:
    diagnosis = result["diagnosis"]
    severity = result["severity"]
    reasoning = result["reasoning"]

    st.divider()
    st.subheader("Diagnostic Result")

    status = diagnosis["status"]
    if status == "DEFECT_DETECTED":
        st.success("Visible defect evidence detected")
    else:
        st.warning("Insufficient evidence for a localized defect")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{diagnosis['confidence']:.0%}")
    m2.metric("Severity", diagnosis["severity"])
    m3.metric("Priority", diagnosis["priority"])
    m4.metric("Review", "Required" if diagnosis["review_required"] else "Not required")

    left, right = st.columns([1.15, 1])
    with left:
        st.image(annotated, caption="Localized evidence", use_container_width=True)
    with right:
        st.markdown("#### Summary")
        st.write(diagnosis["summary"])

        st.markdown("#### Findings")
        if diagnosis["findings"]:
            for finding in diagnosis["findings"]:
                st.write(
                    f"**{finding['defect']}** → {finding['component']} "
                    f"({finding['region']}) · {finding['confidence']:.0%}"
                )
        else:
            st.write("No localized findings.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Possible causes")
        for cause in diagnosis["possible_causes"]:
            st.write(f"• {cause}")
    with c2:
        st.markdown("#### Recommended next actions")
        for action in diagnosis["next_actions"]:
            st.write(f"• {action}")

    with st.expander("Evidence & reasoning", expanded=False):
        for item in reasoning["evidence"]:
            st.write(f"• {item}")
        st.markdown("**Observations**")
        for item in reasoning["observations"]:
            st.write(f"• {item}")
        st.markdown("**Severity rationale**")
        for item in severity["rationale"]:
            st.write(f"• {item}")

    with st.expander("Safety limitations", expanded=False):
        for item in diagnosis["limitations"]:
            st.write(f"• {item}")


st.set_page_config(
    page_title="ElectroDiagnose",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔧 ElectroDiagnose")
st.caption("AI-assisted visual and multimodal diagnostic dashboard")

with st.sidebar:
    st.header("Analysis Settings")
    device = st.selectbox("Device category", DEVICE_OPTIONS)
    mode = st.radio(
        "Inference mode",
        ["Pipeline demo", "YOLO model"],
        help=(
            "Pipeline demo uses deterministic sample detections for a reliable "
            "hackathon presentation. YOLO model runs the configured Ultralytics model."
        ),
    )
    model_path = st.text_input(
        "YOLO model path",
        value="yolo11n.pt",
        disabled=mode != "YOLO model",
    )
    confidence = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)

    st.divider()
    st.markdown("**Pipeline**")
    st.write("Image → Detection → Component → Localization → Reasoning → Severity → Diagnosis")
    st.divider()
    st.markdown("**Recent diagnosis history**")
    for record in list_diagnoses(5):
        st.caption(f"#{record['id']} · {record['device']} · {record['status']} · {record['confidence']:.0%}")

uploaded_files = st.file_uploader(
    "Upload device / component evidence",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="You can upload multiple views of the same device.",
)

col1, col2 = st.columns(2)
with col1:
    symptoms = st.text_area(
        "Symptoms (optional)",
        placeholder="Example: Device powers on but the screen flickers.",
        height=120,
    )
with col2:
    ocr_text = st.text_area(
        "Error / OCR text (optional)",
        placeholder="Paste visible error text, diagnostic output, or log message.",
        height=120,
    )

analyze = st.button("🔍 Analyze Evidence", type="primary", use_container_width=True)

if analyze:
    if not uploaded_files:
        st.warning("Upload at least one image before analysis.")
    else:
        all_locations: list[dict[str, Any]] = []
        all_detections: list[dict[str, Any]] = []
        all_reasoning_evidence: list[str] = []
        last_image: Image.Image | None = None

        progress = st.progress(0.0, text="Preparing evidence...")
        for index, uploaded in enumerate(uploaded_files, start=1):
            image = ImageOps.exif_transpose(Image.open(BytesIO(uploaded.getvalue()))).convert("RGB")
            last_image = image

            if mode == "Pipeline demo":
                detections = _demo_detections(device, image)
            else:
                with st.spinner(f"Running YOLO on {uploaded.name}..."):
                    detections = _run_yolo(
                        device,
                        image,
                        model_path,
                        confidence,
                        iou,
                    )

            result = analyze_evidence(
                device=device,
                image=image,
                detections=detections,
                ocr_text=ocr_text,
                symptoms=symptoms,
            )
            all_detections.extend(result["detections"])
            all_locations.extend(result["locations"])
            all_reasoning_evidence.extend(result["reasoning"]["evidence"])
            progress.progress(
                index / len(uploaded_files),
                text=f"Processed {index}/{len(uploaded_files)} evidence image(s)",
            )

        if last_image is not None:
            combined_reasoning = reason_about_evidence(
                device=device,
                locations=all_locations,
                ocr_text=ocr_text,
                symptoms=symptoms,
            )
            combined_severity = score_severity(all_locations, combined_reasoning)
            combined_diagnosis = diagnose(
                device=device,
                locations=all_locations,
                reasoning=combined_reasoning,
                severity=combined_severity,
            )
            combined = {
                "detections": all_detections,
                "locations": all_locations,
                "reasoning": combined_reasoning,
                "severity": combined_severity,
                "diagnosis": combined_diagnosis,
            }
            annotated = draw_findings(last_image, all_locations)
            if mode == "Pipeline demo":
                st.info("Demo mode: findings are deterministic presentation data, not model predictions.")
            else:
                st.info("YOLO mode: results depend on the selected model and its trained classes.")
            _show_result(combined, annotated)

st.divider()
st.caption(
    "ElectroDiagnose — Hackathon Prototype | "
    "Visual evidence does not by itself prove an internal electrical fault."
)
