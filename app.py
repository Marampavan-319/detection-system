"""Streamlit product dashboard for ElectroDiagnose."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4
import sqlite3

import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from src.components import identify_components
from src.database import authenticate_user, create_user, get_diagnosis, get_user, list_diagnoses, save_diagnosis, update_user
from src.diagnosis import diagnose
from src.localization import localize_defects
from src.reasoning import reason_about_evidence
from src.report import generate_pdf_report
from src.severity import score_severity


DEVICE_OPTIONS = ["Laptop", "Smartphone", "PCB", "Router"]
NAV_ITEMS = ["Dashboard", "AI Advisor", "Analytics", "History", "Profile"]

# Device-specific model slots. The existing Laptop flow keeps its current
# yolo11n.pt default; Smartphone gets its own model path and never reuses the
# Laptop model accidentally. Additional device models can be added later.
DEVICE_MODEL_DEFAULTS = {
    "Laptop": "models/laptop_yolo11n.pt",
    "Smartphone": "models/smartphone_yolo11n.pt",
    "PCB": "yolo11n.pt",
    "Router": "yolo11n.pt",
}


def _default_model_path(device: str) -> str:
    return DEVICE_MODEL_DEFAULTS.get(device, "yolo11n.pt")



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
    return [{
        "class_id": 0,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": [x1 * width, y1 * height, x2 * width, y2 * height],
    }]


def _run_yolo(device: str, image: Image.Image, model_path: str, confidence: float, iou: float):
    """Run the existing YOLO wrapper without importing Ultralytics at app startup."""
    from src.detection import YOLODetector

    detector = YOLODetector(model_path=model_path, confidence=confidence, iou=iou)
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
    result = diagnose(device=device, locations=locations, reasoning=reasoning, severity=severity)
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
        label = f"{index}. {item['defect_class']} | {item['confidence']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        text_box = draw.textbbox((x1, max(0, y1 - 22)), label)
        draw.rectangle(text_box, fill="red")
        draw.text((x1 + 4, max(0, y1 - 20)), label, fill="white")
    return canvas


def _render_history(container, limit: int = 5) -> None:
    """Render recent diagnoses with controls for opening saved records."""
    with container.container():
        st.markdown("**Recent diagnosis history**")
        records = list_diagnoses(limit)
        if not records:
            st.caption("No saved diagnoses yet.")
            return
        render_token = st.session_state.get("history_render_token", 0)
        for record in records:
            col_text, col_button = st.columns([3, 1])
            with col_text:
                st.caption(
                    f"#{record['id']} · {record['device']} · "
                    f"{record['status']} · {record['confidence']:.0%}"
                )
            with col_button:
                if st.button("View", key=f"view_diagnosis_{record['id']}_{render_token}"):
                    st.session_state["selected_diagnosis_id"] = record["id"]
                    st.session_state["page"] = "History"


def _show_result(
    result: dict[str, Any],
    annotated: Image.Image | None,
    history_container,
    evidence_images: list[Image.Image] | None = None,
    symptoms: str = "",
    ocr_text: str = "",
) -> None:
    diagnosis = result["diagnosis"]
    severity = result["severity"]
    reasoning = result["reasoning"]

    st.divider()
    st.subheader("📋 Overall Diagnostic Report")

    if diagnosis["status"] == "DEFECT_DETECTED":
        st.error("🔴 Defect evidence detected")
    else:
        st.warning("🟡 Insufficient evidence for a localized defect")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{diagnosis['confidence']:.0%}")
    m2.metric("Severity", diagnosis["severity"])
    m3.metric("Priority", diagnosis["priority"])
    m4.metric("Review", "Required" if diagnosis["review_required"] else "Not required")

    left, right = st.columns([1.15, 1])
    with left:
        if annotated is not None:
            st.image(annotated, caption="Combined localized evidence", use_container_width=True)
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

    if st.button("Save diagnosis", key="save_diagnosis"):
        image_dir = Path("uploads") / "diagnoses" / uuid4().hex
        image_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        for index, evidence_image in enumerate(evidence_images or [], start=1):
            image_path = image_dir / f"evidence_{index}.png"
            evidence_image.save(image_path, format="PNG")
            image_paths.append(str(image_path))

        if annotated is not None:
            annotated_path = image_dir / "annotated.png"
            annotated.save(annotated_path, format="PNG")
            image_paths.append(str(annotated_path))

        if st.session_state.get("demo_mode", False):
            record_id = len(st.session_state.get("demo_records", [])) + 1
            demo_record = dict(diagnosis)
            demo_record["id"] = record_id
            demo_record["timestamp"] = "Demo session"
            demo_record["evidence"] = {
                "images": image_paths,
                "reasoning": reasoning,
                "symptoms": symptoms,
                "ocr_text": ocr_text,
            }
            st.session_state["demo_records"].insert(0, demo_record)
        else:
            record_id = save_diagnosis(
                diagnosis,
                evidence={
                    "images": image_paths,
                    "reasoning": reasoning,
                    "symptoms": symptoms,
                    "ocr_text": ocr_text,
                },
            )
        st.session_state["last_saved_diagnosis_id"] = record_id
        st.session_state["history_render_token"] += 1
        st.session_state["pending_case_id"] = None
        st.success(f"Diagnosis saved to history (ID {record_id}).")
        _render_history(history_container)

    pdf_bytes = generate_pdf_report(diagnosis)
    st.download_button(
        "PDF report",
        data=pdf_bytes,
        file_name="electrodiagnose_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _show_saved_diagnosis(selected_id: int) -> None:
    selected = get_diagnosis(selected_id)
    if selected is None:
        st.warning("Saved diagnosis could not be found.")
        return

    st.subheader(f"Saved Diagnosis #{selected['id']}")
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Device", selected["device"].title())
    sm2.metric("Confidence", f"{selected['confidence']:.0%}")
    sm3.metric("Severity", selected["severity"])
    sm4.metric("Priority", selected["priority"])
    st.write(f"**Status:** {selected['status']}")
    st.write(f"**Saved:** {selected['timestamp']}")
    st.markdown("**Summary**")
    st.write(selected["summary"])

    st.markdown("**Findings**")
    if selected["findings"]:
        for finding in selected["findings"]:
            st.write(
                f"• **{finding.get('defect', 'Unknown')}** → "
                f"{finding.get('component', 'Unknown')} "
                f"({finding.get('region', 'Unknown')}) · "
                f"{float(finding.get('confidence', 0)):.0%}"
            )
    else:
        st.write("No localized findings.")

    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Possible causes**")
        for cause in selected["possible_causes"]:
            st.write(f"• {cause}")
    with rc2:
        st.markdown("**Recommended next actions**")
        for action in selected["next_actions"]:
            st.write(f"• {action}")

    with st.expander("Evidence & reasoning", expanded=True):
        evidence = selected.get("evidence", {})
        for item in evidence.get("evidence", []):
            st.write(f"• {item}")
        st.markdown("**Observations**")
        for item in evidence.get("observations", []):
            st.write(f"• {item}")
        if evidence.get("symptoms"):
            st.markdown("**Symptoms**")
            st.write(evidence["symptoms"])
        if evidence.get("ocr_text"):
            st.markdown("**Error / OCR text**")
            st.write(evidence["ocr_text"])

    image_paths = selected.get("evidence", {}).get("images", [])
    if image_paths:
        st.markdown("**Evidence images**")
        cols = st.columns(min(3, len(image_paths)))
        for index, image_path in enumerate(image_paths):
            path = Path(image_path)
            if path.exists():
                cols[index % len(cols)].image(
                    str(path),
                    caption="Annotated result" if path.name == "annotated.png" else f"Evidence {index + 1}",
                    use_container_width=True,
                )

    with st.expander("Safety precautions & limitations", expanded=True):
        for item in selected["limitations"]:
            st.write(f"• {item}")

    pdf_bytes = generate_pdf_report(selected)
    st.download_button(
        "PDF report",
        data=pdf_bytes,
        file_name=f"electrodiagnose_report_{selected['id']}.pdf",
        mime="application/pdf",
        use_container_width=True,
        key=f"saved_pdf_{selected['id']}",
    )




def _render_pie_chart(counts: dict[str, int], title: str) -> None:
    """Render a lightweight donut chart without pandas/Plotly dependencies."""
    total = sum(counts.values())
    if total <= 0:
        st.info("No data available.")
        return

    palette = ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
    stops = []
    legend = []
    start = 0.0

    for index, (label, value) in enumerate(counts.items()):
        percent = (value / total) * 100
        end = start + percent
        color = palette[index % len(palette)]
        stops.append(f"{color} {start:.2f}% {end:.2f}%")
        legend.append(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<span style="width:12px;height:12px;border-radius:50%;background:{color};display:inline-block;"></span>'
            f'<span>{label}: {value} ({percent:.0f}%)</span></div>'
        )
        start = end

    chart = (
        '<div style="display:flex;align-items:center;gap:28px;'
        'padding:18px;border:1px solid rgba(128,128,128,.25);'
        'border-radius:12px;margin:8px 0 18px 0;">'
        f'<div style="width:190px;height:190px;border-radius:50%;'
        f'background:conic-gradient({", ".join(stops)});'
        'display:flex;align-items:center;justify-content:center;flex-shrink:0;">'
        '<div style="width:105px;height:105px;border-radius:50%;'
        'background:var(--background-color,#fff);display:flex;'
        'align-items:center;justify-content:center;text-align:center;'
        'font-weight:700;font-size:15px;">'
        f'{total}<br><span style="font-size:11px;font-weight:400;">total cases</span>'
        '</div></div>'
        f'<div><div style="font-size:18px;font-weight:700;margin-bottom:8px;">{title}</div>'
        + "".join(legend)
        + "</div></div>"
    )
    st.markdown(chart, unsafe_allow_html=True)

def _dashboard(records: list[dict[str, Any]]) -> None:
    st.markdown(
        '<div class="hogwarts-brand"><span class="crest">⚡</span><span class="name">HOGWARTS LEGACY</span><span class="edition">5.0</span></div>',
        unsafe_allow_html=True,
    )

    st.title("👋 Welcome to ElectroDiagnose")
    st.caption("AI-assisted visual and multimodal diagnostic dashboard")

    total = len(records)
    defects = sum(r["status"] == "DEFECT_DETECTED" for r in records)
    high = sum(r["severity"] == "HIGH" for r in records)
    medium = sum(r["severity"] == "MEDIUM" for r in records)
    low = sum(r["severity"] == "LOW" for r in records)

    st.markdown("### Overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total diagnoses", total)
    k2.metric("Defects detected", defects)
    k3.metric("High severity", high)
    k4.metric("Medium / Low", medium + low)

    st.markdown("### Device distribution")
    if records:
        device_counts = {}
        for record in records:
            name = record["device"].title()
            device_counts[name] = device_counts.get(name, 0) + 1
        _render_pie_chart(device_counts, "Share of saved cases by device")
        st.caption("This chart shows the share of saved cases by device, not the percentage of devices that are defective.")
    else:
        st.info("Save a diagnosis to populate dashboard analytics.")

    st.markdown("### Recent diagnoses")
    if not records:
        st.info("No saved diagnoses yet. Open AI Advisor to run your first analysis.")
        return

    for record in records[:5]:
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.5, 1.5, 1.2, 0.8])
        c1.write(f"**#{record['id']}**")
        c2.write(record["device"].title())
        c3.write(record["status"].replace("_", " ").title())
        c4.write(f"{record['confidence']:.0%} · {record['severity']}")
        if c5.button("View", key=f"dashboard_view_{record['id']}"):
            st.session_state["selected_diagnosis_id"] = record["id"]
            st.session_state["page"] = "History"
            st.rerun()


def _analytics(records: list[dict[str, Any]]) -> None:
    st.title("📊 Analytics")
    st.caption("A simple view of the checks and problems found so far.")

    if not records:
        st.info("No saved diagnoses yet. Run and save a diagnosis to see analytics.")
        return

    status_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    for record in records:
        status = record["status"].replace("_", " ").title()
        device = record["device"].title()
        severity = record["severity"].title()
        status_counts[status] = status_counts.get(status, 0) + 1
        device_counts[device] = device_counts.get(device, 0) + 1
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    total = len(records)
    defects = status_counts.get("Defect Detected", 0)
    high = severity_counts.get("High", 0)
    review = sum(bool(r["review_required"]) for r in records)
    avg_confidence = sum(float(r["confidence"]) for r in records) / total

    st.markdown("### 📌 At a Glance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Checks", total)
    k2.metric("Problems Found", defects)
    k3.metric("Serious Problems", high)
    k4.metric("AI Surety", f"{avg_confidence:.0%}")

    st.markdown("### 📈 What Problems Are We Seeing?")
    a1, a2 = st.columns(2)
    with a1:
        _render_pie_chart(device_counts, "Problems by device")
    with a2:
        _render_pie_chart(status_counts, "Result of the check")

    st.markdown("### ⚠️ How Serious Are the Problems?")
    st.caption(
        "Each device is shown on a fixed risk scale: 0 = Zero Risk · "
        "1–2 = Low Risk · 3–4 = Medium Risk · 5 = High Risk. "
        "The bar uses the highest saved risk level for that device."
    )

    # X-axis = device. Y-axis = fixed risk score from 0 to 5.
    # Map the existing diagnosis severity labels to the requested scale.
    severity_to_risk = {
        "UNKNOWN": 0,
        "LOW": 1,
        "MEDIUM": 3,
        "HIGH": 5,
    }
    device_risk = {device: 0 for device in DEVICE_OPTIONS}
    for record in records:
        device = record["device"].title()
        risk = severity_to_risk.get(record["severity"].upper(), 0)
        device_risk[device] = max(device_risk.get(device, 0), risk)

    st.bar_chart(
        device_risk,
        x_label="Device",
        y_label="Risk level (0–5)",
        height=320,
    )

    st.markdown("### 🔍 How Sure Is the AI?")
    q1, q2, q3 = st.columns(3)
    q1.metric("How Sure the AI Is", f"{avg_confidence:.0%}")
    q2.metric("Needs Another Check", review)
    q3.metric("Not Enough Information", status_counts.get("Insufficient Evidence", 0))

    st.markdown("### 🕘 Recent Cases")
    for record in records[:5]:
        status_label = {"DEFECT_DETECTED": "Problem Found", "INSUFFICIENT_EVIDENCE": "Not Enough Information"}.get(record["status"], record["status"].replace("_", " ").title())
        saved_time = record["timestamp"].replace("T", " ")[:19]
        st.write(
            f"**Case #{record['id']}** · {record['device'].title()} · "
            f"{status_label} · {record['confidence']:.0%} · "
            f"{record['severity']} · {saved_time} UTC"
        )

    st.caption(
        "Analytics are based only on saved diagnoses currently stored in the local SQLite database."
    )


def _history(records: list[dict[str, Any]]) -> None:
    st.title("🕘 Diagnosis History")
    st.caption("Search, filter, and open complete saved diagnostic cases.")

    if not records:
        st.info("No saved diagnoses yet. Run and save a diagnosis from AI Advisor.")
        return

    selected_id = st.session_state.get("selected_diagnosis_id")

    # Case-detail mode: show only the selected case.
    # The full history list stays hidden until the user presses the back arrow.
    if selected_id is not None:
        top_back, _ = st.columns([0.15, 0.85])
        with top_back:
            if st.button("←", key="history_back", help="Back to all cases"):
                st.session_state["selected_diagnosis_id"] = None
                st.session_state["history_case_id"] = ""
                st.rerun()
        _show_saved_diagnosis(selected_id)
        return

    # Summary metrics for the full history list.
    total = len(records)
    defects = sum(r["status"] == "DEFECT_DETECTED" for r in records)
    high = sum(r["severity"] == "HIGH" for r in records)
    review = sum(bool(r["review_required"]) for r in records)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Saved cases", total)
    m2.metric("Defects detected", defects)
    m3.metric("High severity", high)
    m4.metric("Review required", review)

    st.markdown("### 🔎 Find a case by Case ID")
    st.caption("Type the exact Case ID and press Enter to open it.")

    f1, f2, f3 = st.columns([2.4, 1.1, 1.1])

    with f1:
        case_id_input = st.text_input(
            "Case ID",
            placeholder="Enter Case ID and press Enter",
            label_visibility="collapsed",
            key="history_case_id",
        )

    with f2:
        device_filter = st.selectbox(
            "Device",
            ["All"] + sorted({r["device"].title() for r in records}),
            key="history_device_filter",
        )

    with f3:
        severity_filter = st.selectbox(
            "Severity",
            ["All", "HIGH", "MEDIUM", "LOW", "UNKNOWN"],
            key="history_severity_filter",
        )

    status_filter = st.selectbox(
        "Status filter",
        ["All", "DEFECT_DETECTED", "INSUFFICIENT_EVIDENCE"],
        key="history_status_filter",
    )

    # Streamlit text_input triggers a rerun when Enter is pressed.
    # Search automatically whenever a non-empty Case ID is entered.
    case_id_text = case_id_input.strip()
    if case_id_text:
        if not case_id_text.isdigit():
            st.warning("Case ID must be a number, for example 12.")
        else:
            requested_id = int(case_id_text)
            selected_case = get_diagnosis(requested_id)
            if selected_case is None:
                st.error(f"No saved case found with Case ID #{requested_id}.")
                st.session_state["selected_diagnosis_id"] = None
            elif st.session_state.get("selected_diagnosis_id") != requested_id:
                st.session_state["selected_diagnosis_id"] = requested_id
                st.rerun()

    filtered = []
    for record in records:
        if device_filter != "All" and record["device"].title() != device_filter:
            continue
        if severity_filter != "All" and record["severity"] != severity_filter:
            continue
        if status_filter != "All" and record["status"] != status_filter:
            continue
        filtered.append(record)

    st.caption(f"Showing {len(filtered)} of {total} saved case(s).")

    if not filtered:
        st.info("No cases match the current search and filters.")
        return

    st.markdown("### 📋 Cases")

    for record in filtered:
        status_label = {"DEFECT_DETECTED": "Problem Found", "INSUFFICIENT_EVIDENCE": "Not Enough Information"}.get(record["status"], record["status"].replace("_", " ").title())
        severity_label = record["severity"].title()
        priority_label = record["priority"].title()
        saved_time = record["timestamp"].replace("T", " ")[:19]

        with st.container(border=True):
            top = st.columns([0.75, 1.15, 1.45, 1.05, 1.0, 0.8])
            top[0].markdown(f"**#{record['id']}**")
            top[1].write(record["device"].title())
            top[2].write(status_label)
            top[3].write(f"{record['confidence']:.0%}")
            top[4].write(f"{severity_label} / {priority_label}")
            if top[5].button("View", key=f"history_view_{record['id']}"):
                st.session_state["selected_diagnosis_id"] = record["id"]
                st.rerun()

            st.caption(f"Saved: {saved_time} UTC")
            if record.get("summary"):
                st.write(record["summary"])


def _profile(records: list[dict[str, Any]]) -> None:
    st.title("👤 Profile")
    st.caption("Your ElectroDiagnose profile and usage summary.")

    total = len(records)
    defects = sum(r["status"] == "DEFECT_DETECTED" for r in records)
    high = sum(r["severity"] == "HIGH" for r in records)
    avg_confidence = (
        sum(float(r["confidence"]) for r in records) / total
        if total
        else 0.0
    )

    if "profile_name" not in st.session_state:
        st.session_state["profile_name"] = "ElectroDiagnose User"
    if "profile_role" not in st.session_state:
        st.session_state["profile_role"] = "Diagnostic user"
    if "profile_number" not in st.session_state:
        st.session_state["profile_number"] = ""
    if "profile_email" not in st.session_state:
        st.session_state["profile_email"] = ""
    if "profile_editing" not in st.session_state:
        st.session_state["profile_editing"] = False

    st.markdown("### 👤 Profile Information")

    if st.session_state["profile_editing"]:
        with st.container(border=True):
            st.markdown("#### ✏️ Edit Profile")

            edited_name = st.text_input(
                "Name",
                value=st.session_state["profile_name"],
                key="edit_profile_name",
            )
            edited_role = st.text_input(
                "Role *",
                value=st.session_state["profile_role"],
                key="edit_profile_role",
            )
            edited_number = st.text_input(
                "Phone Number *",
                value=st.session_state["profile_number"],
                key="edit_profile_number",
                placeholder="Enter phone number",
            )
            edited_email = st.text_input(
                "Email *",
                value=st.session_state["profile_email"],
                key="edit_profile_email",
                placeholder="Enter email address",
            )

            ec1, ec2 = st.columns(2)
            with ec1:
                if st.button(
                    "💾 Save Changes",
                    type="primary",
                    use_container_width=True,
                    key="save_profile",
                ):
                    name = edited_name.strip()
                    role = edited_role.strip()
                    number = edited_number.strip()
                    email = edited_email.strip()

                    if not name:
                        st.warning("Please enter your name.")
                    elif not role:
                        st.warning("Please enter your role.")
                    elif not number:
                        st.warning("Please enter your phone number.")
                    elif not email:
                        st.warning("Please enter your email.")
                    elif "@" not in email or "." not in email.rsplit("@", 1)[-1]:
                        st.warning("Please enter a valid email address.")
                    else:
                        st.session_state["profile_name"] = name
                        st.session_state["profile_role"] = role
                        st.session_state["profile_number"] = number
                        st.session_state["profile_email"] = email
                        st.session_state["profile_editing"] = False
                        st.success("Profile updated successfully.")
                        st.rerun()

            with ec2:
                if st.button(
                    "Cancel",
                    use_container_width=True,
                    key="cancel_profile",
                ):
                    st.session_state["profile_editing"] = False
                    st.rerun()
    else:
        p1, p2 = st.columns(2)

        with p1:
            with st.container(border=True):
                st.markdown("#### User")
                st.markdown(f"**{st.session_state['profile_name']}**")
                st.caption("Hackathon prototype account")
                st.write(f"Role: {st.session_state['profile_role']}")
                st.write(f"Phone: {st.session_state['profile_number'] or 'Not set'}")
                st.write(f"Email: {st.session_state['profile_email'] or 'Not set'}")

                if st.button(
                    "✏️ Edit Profile",
                    use_container_width=True,
                    key="edit_profile",
                ):
                    st.session_state["profile_editing"] = True
                    st.rerun()

        with p2:
            with st.container(border=True):
                st.markdown("#### System")
                st.markdown("**ElectroDiagnose**")
                st.caption("AI-assisted visual defect analysis")
                st.write("Diagnosis history: Local SQLite database")

    st.markdown("### 📊 My Usage")
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Total Checks", total)
    u2.metric("Problems Found", defects)
    u3.metric("Serious Problems", high)
    u4.metric("AI Surety", f"{avg_confidence:.0%}")

    st.markdown("### 🔧 What ElectroDiagnose Does")
    st.info(
        "ElectroDiagnose checks uploaded device images and other evidence, "
        "then provides possible defects, possible causes, recommended next "
        "actions, confidence, severity, and safety limitations."
    )

    st.markdown("### 🔐 Account")
    st.caption(
        "Login and account authentication will be connected in the final "
        "remodel stage."
    )

def _ai_detector() -> None:
    st.title("🤖 AI Advisor")
    st.caption("Manage diagnostic conversations and run new evidence-based analyses.")

    chat_mode = st.radio(
        "AI Detector", ["New Chat", "Old Chats"], horizontal=True, label_visibility="collapsed",
    )

    if chat_mode == "Old Chats":
        st.markdown("### 💬 Old Chats")
        st.caption("Previously saved diagnostic cases. Open any case to review the complete result.")
        records = list_diagnoses(100)
        if not records:
            st.info("No old chats yet. Start a New Chat and save a diagnosis.")
            return
        selected_chat_id = st.session_state.get("selected_chat_id")
        if selected_chat_id is not None:
            selected = get_diagnosis(selected_chat_id)
            if selected is not None:
                if st.button("← Back to Old Chats", key="old_chats_back"):
                    st.session_state["selected_chat_id"] = None
                    st.rerun()
                _show_saved_diagnosis(selected_chat_id)
                return
            st.session_state["selected_chat_id"] = None
        for record in records:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([1.0, 1.4, 2.0, 1.0])
                c1.write(f"**Chat #{record['id']}**")
                c2.write(record["device"].title())
                c3.write(f"{record['status'].replace('_', ' ').title()} · {record['severity']} · {record['confidence']:.0%}")
                if c4.button("Open", key=f"old_chat_{record['id']}"):
                    st.session_state["selected_chat_id"] = record["id"]
                    st.rerun()
        return

    st.markdown("### ✨ New Chat")
    st.caption("Start a fresh diagnostic case using one or more evidence images.")
    device = st.selectbox(
        "Device category", DEVICE_OPTIONS,
        help="Choose the device type first. Laptop and Smartphone use their category-specific detector when installed.",
    )

    use_trained_model = device in {"Laptop", "Smartphone"}
    model_path = _default_model_path(device)
    inference_confidence = 0.25
    inference_iou = 0.45

    if use_trained_model:
        device_icon = "💻" if device == "Laptop" else "📱"
        st.info(f"{device_icon} **{device} AI Advisor** · Using trained model slot: {model_path}")
        if not Path(model_path).exists():
            st.warning(
                f"{device_icon} The trained {device} detector is not installed at {model_path}. "
                "The old hard-coded demo detection has been disabled so the app will not report a fake fixed confidence."
            )
    else:
        st.info(f"🤖 **{device} AI Advisor** · Existing presentation pipeline preserved.")

    st.caption("Confidence values are produced automatically by the active detection pipeline; you do not need to set confidence or IoU manually.")
    st.markdown("**Pipeline**")
    st.write("Image → Device Detector → Component → Localization → Reasoning → Severity → Diagnosis")
    history_container = st.empty()

    uploaded_files = st.file_uploader(
        "Upload device / component evidence",
        type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
        help="Upload multiple views of the same device. Every uploaded image is analyzed separately.",
    )

    col1, col2 = st.columns(2)
    with col1:
        symptoms = st.text_area("Symptoms (optional)", placeholder="Example: Device powers on but the screen flickers.", height=120)
    with col2:
        ocr_text = st.text_area("Error / OCR text (optional)", placeholder="Paste visible error text, diagnostic output, or log message.", height=120)

    analyze = st.button("🔍 Analyze Evidence", type="primary", use_container_width=True, key="new_chat_analyze")

    if analyze:
        if not uploaded_files:
            st.warning("Upload at least one image before analysis.")
        else:
            all_locations = []
            all_detections = []
            evidence_results = []
            progress = st.progress(0.0, text="Preparing evidence...")

            for index, uploaded in enumerate(uploaded_files, start=1):
                image = ImageOps.exif_transpose(Image.open(BytesIO(uploaded.getvalue()))).convert("RGB")
                with st.spinner(f"Running {device} AI detector on {uploaded.name}..."):
                    if use_trained_model:
                        if not Path(model_path).exists():
                            detections = []
                        else:
                            detections = _run_yolo(device, image, model_path, inference_confidence, inference_iou)
                    else:
                        detections = _demo_detections(device, image)

                result = analyze_evidence(device=device, image=image, detections=detections, ocr_text=ocr_text, symptoms=symptoms)
                annotated = draw_findings(image, result["locations"])
                evidence_results.append({
                    "name": uploaded.name,
                    "image": image,
                    "annotated": annotated,
                    "detections": result["detections"],
                    "locations": result["locations"],
                    "reasoning": result["reasoning"],
                    "severity": result["severity"],
                    "diagnosis": result["diagnosis"],
                })
                all_detections.extend(result["detections"])
                all_locations.extend(result["locations"])
                progress.progress(index / len(uploaded_files), text=f"Processed {index}/{len(uploaded_files)} evidence image(s)")

            combined_reasoning = reason_about_evidence(device=device, locations=all_locations, ocr_text=ocr_text, symptoms=symptoms)
            combined_severity = score_severity(all_locations, combined_reasoning)
            combined_diagnosis = diagnose(device=device, locations=all_locations, reasoning=combined_reasoning, severity=combined_severity)
            combined = {
                "detections": all_detections, "locations": all_locations,
                "reasoning": combined_reasoning, "severity": combined_severity,
                "diagnosis": combined_diagnosis, "evidence_results": evidence_results,
                "image_count": len(evidence_results),
            }

            if use_trained_model and Path(model_path).exists():
                st.success(f"{'💻' if device == 'Laptop' else '📱'} {device} trained detector completed across all {len(evidence_results)} uploaded image(s).")
            elif use_trained_model:
                st.error(f"No trained {device} model is installed, so no model detections were generated. Install the category-specific model before testing.")
            else:
                st.info(f"🤖 {device} existing presentation pipeline completed across all {len(evidence_results)} uploaded image(s).")

            st.session_state["latest_result"] = combined
            st.session_state["latest_annotated"] = evidence_results[-1]["annotated"] if evidence_results else None
            st.session_state["latest_evidence_images"] = [item["image"] for item in evidence_results]
            st.session_state["latest_evidence_results"] = evidence_results
            st.session_state["latest_symptoms"] = symptoms
            st.session_state["latest_ocr_text"] = ocr_text

    if st.session_state["latest_result"] is not None:
        latest_result = st.session_state["latest_result"]
        evidence_results = st.session_state.get("latest_evidence_results", [])

        if evidence_results:
            st.markdown("### 🖼️ Individual Image Analysis")
            st.caption(
                f"{len(evidence_results)} of {len(evidence_results)} uploaded image(s) were analyzed independently."
            )

            for index, item in enumerate(evidence_results, start=1):
                image_diagnosis = item["diagnosis"]
                image_reasoning = item["reasoning"]

                with st.container(border=True):
                    st.markdown(f"#### Image {index} — {item['name']}")
                    image_col, detail_col = st.columns([1, 2.5])

                    with image_col:
                        st.image(
                            item["annotated"],
                            caption=(
                                f"{len(item['locations'])} localized finding(s)"
                                if item["locations"]
                                else "No localized findings"
                            ),
                            width=230,
                        )

                    with detail_col:
                        im1, im2, im3 = st.columns(3)
                        im1.metric("Confidence", f"{image_diagnosis['confidence']:.0%}")
                        im2.metric("Severity", image_diagnosis["severity"])
                        im3.metric(
                            "Result",
                            "Defect detected"
                            if image_diagnosis["status"] == "DEFECT_DETECTED"
                            else "Insufficient evidence",
                        )

                        st.markdown("**Explanation**")
                        st.write(image_diagnosis["summary"])

                        if image_diagnosis["findings"]:
                            st.markdown("**Findings**")
                            for finding in image_diagnosis["findings"]:
                                st.write(
                                    f"• **{finding['defect']}** → {finding['component']} "
                                    f"({finding['region']}) · {finding['confidence']:.0%}"
                                )
                        else:
                            st.write("No localized findings were identified in this image.")

                        if image_reasoning["evidence"]:
                            st.markdown("**Evidence observed**")
                            for evidence_item in image_reasoning["evidence"][:3]:
                                st.write(f"• {evidence_item}")

                        if image_diagnosis["next_actions"]:
                            st.markdown("**Recommended action**")
                            st.write(image_diagnosis["next_actions"][0])

        st.markdown("### 📊 Overall Report")
        st.caption(
            "This report combines the evidence from every uploaded image; it is not based only on the last image."
        )

        _show_result(
            latest_result,
            None,
            history_container,
            st.session_state.get("latest_evidence_images", []),
            st.session_state.get("latest_symptoms", ""),
            st.session_state.get("latest_ocr_text", ""),
        )

def _login_screen() -> None:
    """Render login and account creation before exposing the application."""
    st.markdown(
        '<div class="login-shell"><div class="login-card">'+
        '<div class="login-event">⚡ HOGWARTS LEGACY <span>5.0</span></div>'+
        '<div class="login-title">Welcome to ElectroDiagnose</div>'+
        '<div class="login-subtitle">AI-powered defect analysis for devices and electronics</div>'+
        '</div></div>',
        unsafe_allow_html=True,
    )

    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "Login"

    mode = st.session_state["auth_mode"]

    if mode == "Login":
        form_col_left, form_col, form_col_right = st.columns([1.5, 1, 1.5])
        with form_col:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("🔐 Login", use_container_width=True, type="primary")
                switch_to_create = st.form_submit_button("Create Account", use_container_width=True)

            demo_login = st.button("🎯 Demo Account", use_container_width=True, key="demo_login")
        if demo_login:
            st.session_state["authenticated"] = True
            st.session_state["demo_mode"] = True
            st.session_state["user"] = {
                "id": "demo",
                "name": "Demo User",
                "role": "Judge / Demo",
                "phone": "Demo",
                "email": "demo@electrodiagnose.local",
            }
            st.session_state["page"] = "Dashboard"
            st.session_state["profile_name"] = "Demo User"
            st.session_state["profile_role"] = "Judge / Demo"
            st.session_state["profile_number"] = "Demo"
            st.session_state["profile_email"] = "demo@electrodiagnose.local"
            st.session_state["demo_records"] = []
            st.rerun()
        if switch_to_create:
            st.session_state["auth_mode"] = "Create Account"
            st.rerun()
        if submitted:
            if not email.strip() or not password:
                st.error("Please enter your email and password.")
            else:
                user = authenticate_user(email, password)
                if user is None:
                    st.error("Email or password is incorrect.")
                else:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user
                    st.session_state["page"] = "Dashboard"
                    st.session_state["profile_name"] = user["name"]
                    st.session_state["profile_role"] = user["role"]
                    st.session_state["profile_number"] = user["phone"]
                    st.session_state["profile_email"] = user["email"]
                    st.rerun()
    else:
        form_col_left, form_col, form_col_right = st.columns([1.5, 1, 1.5])
        with form_col:
            with st.form("create_account_form"):
                name = st.text_input("Full Name")
                role = st.text_input("Role")
                phone = st.text_input("Phone Number")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                switch_to_login = st.form_submit_button("Login", use_container_width=True)
        if switch_to_login:
            st.session_state["auth_mode"] = "Login"
            st.rerun()
        if submitted:
            if not all([name.strip(), role.strip(), phone.strip(), email.strip(), password, confirm]):
                st.error("Please fill in all required fields.")
            elif "@" not in email or "." not in email.rsplit("@", 1)[-1]:
                st.error("Please enter a valid email address.")
            elif len(password) < 6:
                st.error("Password must contain at least 6 characters.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    user_id = create_user(name, role, phone, email, password)
                except sqlite3.IntegrityError:
                    st.error("An account with this email already exists.")
                else:
                    user = get_user(user_id)
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user
                    st.session_state["page"] = "Dashboard"
                    st.session_state["profile_name"] = user["name"]
                    st.session_state["profile_role"] = user["role"]
                    st.session_state["profile_number"] = user["phone"]
                    st.session_state["profile_email"] = user["email"]
                    st.rerun()

st.set_page_config(
    page_title="ElectroDiagnose",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Professional HOGWARTS LEGACY light theme ---
st.markdown(
    """
    <style>
    :root {
        --hogwarts-bg: #f5f9ff;
        --hogwarts-panel: #ffffff;
        --hogwarts-panel-2: #eef6ff;
        --hogwarts-border: #d9e7f5;
        --hogwarts-text: #18324a;
        --hogwarts-muted: #64748b;
        --hogwarts-blue: #2563eb;
        --hogwarts-blue-light: #dbeafe;
        --hogwarts-green: #16a34a;
        --hogwarts-green-light: #dcfce7;
    }

    .stApp {
        background: linear-gradient(135deg, #f7fbff 0%, #eef7ff 55%, #f0fdf4 100%);
        color: var(--hogwarts-text);
    }

    [data-testid="stHeader"] {
        display: flex !important;
    }

    /* Keep the navigation sidebar permanently visible. */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: translateX(0) !important;
        width: 336px !important;
        min-width: 336px !important;
        max-width: 336px !important;
    }

    /* Hide only the collapsed-sidebar overlay/control, never the sidebar itself. */
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }


    [data-testid="stToolbar"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden !important;
    }

    footer {
        visibility: hidden !important;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--hogwarts-border);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: var(--hogwarts-text);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #173b63 !important;
        letter-spacing: -0.02em;
    }

    p, label, .stCaption {
        color: var(--hogwarts-muted);
    }

    .login-shell {
        width: min(92vw, 620px);
        margin: 8vh auto 8vh auto;
        padding: 0 28px;
        box-sizing: border-box;
    }

    .login-card {
        width: 100%;
        max-width: 560px;
        margin: 0 auto;
        text-align: center;
        padding: 34px 34px 24px 34px;
        box-sizing: border-box;
        border: 1px solid #dbeafe;
        border-radius: 22px;
        background: rgba(255, 255, 255, 0.97);
        box-shadow: 0 14px 40px rgba(30, 64, 175, 0.12);
    }

    /* Authentication forms share the exact same centered column as the Demo Account button. */
    div[data-testid="stForm"] {
        width: 100%;
        margin: 28px 0 20px 0;
        padding: 28px 30px 26px 30px;
        border: 1px solid #dbeafe;
        border-radius: 20px;
        background: #ffffff;
        box-shadow: 0 12px 32px rgba(30, 64, 175, 0.10);
        box-sizing: border-box;
    }

        .login-event {
        display: inline-block;
        padding: 8px 14px;
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        background: linear-gradient(90deg, #eff6ff, #f0fdf4);
        color: #1d4ed8;
        font-size: 14px;
        font-weight: 800;
        letter-spacing: 0.04em;
    }

    .login-event span {
        margin-left: 6px;
        padding: 2px 7px;
        border-radius: 999px;
        background: #dcfce7;
        color: #15803d;
        font-size: 11px;
    }

    .login-title {
        margin-top: 18px;
        color: #173b63;
        font-size: 30px;
        font-weight: 800;
    }

    .login-subtitle {
        margin-top: 6px;
        color: #64748b;
        font-size: 14px;
    }

    .hogwarts-global-brand {
        position: fixed;
        top: 12px;
        left: 18px;
        z-index: 999999;
        display: flex;
        align-items: center;
        gap: 9px;
        padding: 8px 13px;
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.10);
    }

    .hogwarts-global-brand .crest {
        font-size: 18px;
    }

    .hogwarts-global-brand .name {
        font-size: 15px;
        font-weight: 800;
        color: #1d4ed8;
        letter-spacing: 0.04em;
    }

    .hogwarts-global-brand .edition {
        padding: 2px 7px;
        border-radius: 999px;
        background: #dcfce7;
        color: #15803d;
        font-size: 11px;
        font-weight: 700;
    }

    .hogwarts-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 18px;
        padding: 10px 16px;
        width: fit-content;
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        background: linear-gradient(90deg, #eff6ff, #f0fdf4);
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
    }

    .hogwarts-brand .crest {
        font-size: 22px;
    }

    .hogwarts-brand .name {
        font-size: 18px;
        font-weight: 800;
        color: #1d4ed8;
        letter-spacing: 0.04em;
    }

    .hogwarts-brand .edition {
        padding: 3px 8px;
        border-radius: 999px;
        background: #dcfce7;
        color: #15803d;
        font-size: 12px;
        font-weight: 700;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--hogwarts-border);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 5px 18px rgba(30, 64, 175, 0.07);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #174a7c !important;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #bfdbfe;
        background: #ffffff;
        color: #1e40af;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #60a5fa;
        background: #eff6ff;
        color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #16a34a);
        color: white;
        border: none;
    }

    .stTextInput > div > div,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextArea > div > div {
        background: #ffffff;
        border-color: #cbdff2;
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #93c5fd;
        border-radius: 14px;
        padding: 8px;
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid var(--hogwarts-border);
        border-radius: 12px;
    }

    hr {
        border-color: #dbe7f3;
    }

    .stAlert {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hogwarts-global-brand"><span class="crest">⚡</span><span class="name">HOGWARTS LEGACY</span><span class="edition">5.0</span></div>',
    unsafe_allow_html=True,
)


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "demo_mode" not in st.session_state:
    st.session_state["demo_mode"] = False
if "demo_records" not in st.session_state:
    st.session_state["demo_records"] = []

if "latest_result" not in st.session_state:
    st.session_state["latest_result"] = None
if "latest_annotated" not in st.session_state:
    st.session_state["latest_annotated"] = None
if "latest_evidence_images" not in st.session_state:
    st.session_state["latest_evidence_images"] = []
if "latest_symptoms" not in st.session_state:
    st.session_state["latest_symptoms"] = ""
if "latest_ocr_text" not in st.session_state:
    st.session_state["latest_ocr_text"] = ""
if "pending_case_id" not in st.session_state:
    st.session_state["pending_case_id"] = None
if "history_render_token" not in st.session_state:
    st.session_state["history_render_token"] = 0
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

if not st.session_state["authenticated"]:
    _login_screen()
    st.stop()

with st.sidebar:
    st.title("🔧 ElectroDiagnose")
    st.caption("AI-powered defect analysis")
    st.divider()
    for item in NAV_ITEMS:
        if st.button(item, key=f"nav_{item.lower().replace(' ', '_')}", use_container_width=True):
            st.session_state["page"] = item
            if item != "History":
                st.session_state["selected_diagnosis_id"] = None
            st.rerun()
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.session_state["demo_mode"] = False
        st.session_state["demo_records"] = []
        st.session_state["latest_result"] = None
        st.session_state["latest_annotated"] = None
        st.session_state["latest_evidence_images"] = []
        st.session_state["latest_symptoms"] = ""
        st.session_state["latest_ocr_text"] = ""
        st.session_state["selected_diagnosis_id"] = None
        st.session_state["selected_chat_id"] = None
        st.session_state["page"] = "Dashboard"
        st.rerun()

records = (
    st.session_state.get("demo_records", [])
    if st.session_state.get("demo_mode", False)
    else list_diagnoses(100)
)
page = st.session_state["page"]

if page == "Dashboard":
    _dashboard(records)
elif page == "AI Advisor":
    _ai_detector()
elif page == "History":
    _history(records)
elif page == "Analytics":
    _analytics(records)
elif page == "Profile":
    _profile(records)

st.divider()
st.caption(
    "ElectroDiagnose — Hackathon Prototype | "
    "Visual evidence does not by itself prove an internal electrical fault."
)