"""PDF report generation for ElectroDiagnose."""

from io import BytesIO


def generate_pdf_report(diagnosis, title="ElectroDiagnose Diagnostic Report"):
    """Return a PDF report as bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    story.append(Paragraph("Device: " + str(diagnosis.get("device", "Unknown")), styles["BodyText"]))
    story.append(Paragraph("Status: " + str(diagnosis.get("status", "Unknown")), styles["BodyText"]))
    story.append(Paragraph("Confidence: {:.0%}".format(float(diagnosis.get("confidence") or 0)), styles["BodyText"]))
    story.append(Paragraph("Severity: " + str(diagnosis.get("severity", "UNKNOWN")), styles["BodyText"]))
    story.append(Paragraph("Priority: " + str(diagnosis.get("priority", "LOW")), styles["BodyText"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(str(diagnosis.get("summary", "")), styles["BodyText"]))
    for heading, key in [("Findings", "findings"), ("Possible causes", "possible_causes"), ("Recommended next actions", "next_actions"), ("Safety limitations", "limitations")]:
        story.append(Spacer(1, 8))
        story.append(Paragraph(heading, styles["Heading2"]))
        for item in diagnosis.get(key, []):
            text = str(item)
            if isinstance(item, dict):
                text = " | ".join(f"{k}: {v}" for k, v in item.items())
            story.append(Paragraph("- " + text, styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()
