import streamlit as st

st.set_page_config(page_title="ElectroDiagnose", page_icon="🔧", layout="wide")

st.title("🔧 ElectroDiagnose")
st.subheader("AI-Powered Electronic Device Diagnostic Assistant")

st.write(
    "Upload the evidence you have. You do not need to know what is wrong "
    "with the device before starting the analysis."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    device_image = st.file_uploader(
        "📷 Device Image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a photo of the complete electronic device.",
    )
    internal_image = st.file_uploader(
        "🔬 Internal / Component Image (Optional)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a PCB, connector, component, or internal-device image.",
    )

with col2:
    error_screenshot = st.file_uploader(
        "🖥️ Error / Software Screenshot (Optional)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload an error message, warning, diagnostic screen, or software screenshot.",
    )
    symptoms = st.text_area(
        "📝 Symptoms (Optional)",
        placeholder="Example: Device turns on, but the screen flickers when the lid is moved.",
        height=120,
    )

st.divider()

if st.button("🔍 Analyze Device", type="primary", use_container_width=True):
    if not any([device_image, internal_image, error_screenshot, symptoms.strip()]):
        st.warning("Please provide at least one image or symptom description.")
    else:
        st.info(
            "Step 1 is complete. The input pipeline is ready. "
            "AI detection and diagnostic reasoning will be connected in later steps."
        )
        input_count = sum([
            device_image is not None,
            internal_image is not None,
            error_screenshot is not None,
            bool(symptoms.strip()),
        ])
        st.markdown("### Current Input")
        st.write(f"Evidence items provided: **{input_count}**")
        st.markdown("### Planned Diagnostic Output")
        st.json({
            "device": "Pending AI analysis",
            "affected_component": "Pending component detection",
            "defect": "Pending defect detection",
            "confidence": None,
            "severity": "Pending",
            "possible_cause": "Pending multimodal reasoning",
            "recommendation": "Pending diagnostic assessment",
        })

st.divider()
st.caption(
    "ElectroDiagnose — Hackathon Prototype | "
    "AI-assisted diagnosis, not a substitute for qualified repair inspection."
)
