import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

try:
    from backend.config import REPORT_PATH, ensure_dirs
except ImportError:
    from config import REPORT_PATH, ensure_dirs  # type: ignore


def _load_metrics() -> Dict[str, object]:
    metrics_path = Path(REPORT_PATH) / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _safe_table(df: pd.DataFrame, cols: list, limit: int) -> str:
    if df.empty:
        return "<p>No records available.</p>"
    return df[cols].head(limit).to_html(index=False, classes="table table-sm")


def generate_detection_report() -> Path:
    ensure_dirs()
    report_dir = Path(REPORT_PATH)
    predictions_path = report_dir / "predictions_master.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    predictions = pd.read_csv(predictions_path)
    metrics = _load_metrics()

    total = len(predictions)
    total_real = int((predictions["actual_label"].str.lower() == "real").sum())
    total_fake = int((predictions["actual_label"].str.lower() == "fake").sum())
    video_count = int(
        predictions["file_path"]
        .str.lower()
        .str.contains(r"\.(mp4|avi|mov|mkv|flv)$", regex=True)
        .sum()
    )
    image_count = total - video_count

    correct_real = predictions[
        (predictions["actual_label"].str.lower() == "real")
        & (predictions["predicted_label"].str.lower() == "real")
    ]
    correct_fake = predictions[
        (predictions["actual_label"].str.lower() == "fake")
        & (predictions["predicted_label"].str.lower() == "fake")
    ]
    misclassified = predictions[
        predictions["actual_label"].str.lower() != predictions["predicted_label"].str.lower()
    ]

    model_section = """
    <ul>
      <li>ResNet-50 accuracy: Refer evaluation split metrics</li>
      <li>EfficientNetB3 accuracy: Refer evaluation split metrics</li>
      <li>ViT accuracy: Refer evaluation split metrics</li>
      <li>Ensemble voting: Majority vote across 3 models</li>
    </ul>
    """

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Veridex Detection Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.5; }}
    h1, h2 {{ color: #1f3c88; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    .metric {{ display: inline-block; min-width: 170px; margin: 6px 12px 6px 0; }}
    img {{ max-width: 700px; border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
    th {{ background: #f5f7fa; }}
  </style>
</head>
<body>
  <h1>Deepfake Detection Report</h1>
  <p>Generated: {datetime.now().isoformat(timespec="seconds")}</p>

  <div class="card">
    <h2>Summary Statistics</h2>
    <div class="metric"><b>Total files processed:</b> {total}</div>
    <div class="metric"><b>Total images processed:</b> {image_count}</div>
    <div class="metric"><b>Total videos processed:</b> {video_count}</div>
    <div class="metric"><b>Real samples:</b> {total_real}</div>
    <div class="metric"><b>Fake samples:</b> {total_fake}</div>
  </div>

  <div class="card">
    <h2>Model Performance</h2>
    <pre>{json.dumps(metrics, indent=2)}</pre>
    <h3>ROC Curve</h3>
    <img src="roc_curve.png" alt="ROC Curve"/>
    <h3>Confusion Matrix</h3>
    <img src="confusion_matrix.png" alt="Confusion Matrix"/>
  </div>

  <div class="card">
    <h2>Per-Model Breakdown</h2>
    {model_section}
  </div>

  <div class="card">
    <h2>Anomaly Detection Statistics</h2>
    <p>
      Frequency, facial landmark, color-space, optical-flow, and lighting anomalies are
      aggregated in predictor output and used as supporting evidence for final classification.
    </p>
  </div>

  <div class="card">
    <h2>Sample Predictions</h2>
    <h3>Correctly Identified Real (Top 5)</h3>
    {_safe_table(correct_real, ["file_path", "confidence"], 5)}
    <h3>Correctly Identified Fake (Top 5)</h3>
    {_safe_table(correct_fake, ["file_path", "confidence"], 5)}
    <h3>Misclassified Examples (Top 3)</h3>
    {_safe_table(misclassified, ["file_path", "actual_label", "predicted_label", "confidence"], 3)}
  </div>
</body>
</html>
"""

    out_file = report_dir / "detection_report.html"
    out_file.write_text(html, encoding="utf-8")
    print(f"Saved HTML report to {out_file}")
    return out_file


def generate_threat_report(report_payload: Dict[str, object]) -> str:
    result = str(report_payload.get("prediction") or report_payload.get("result") or "UNKNOWN")
    confidence = report_payload.get("confidence", 0)
    reasons = report_payload.get("reasoning", [])
    if isinstance(reasons, list):
        reason_text = "; ".join(str(r) for r in reasons[:4]) or "No anomaly indicators."
    else:
        reason_text = str(reasons)
    return (
        f"Threat assessment: {result} at {confidence}% confidence. "
        f"Key indicators: {reason_text}"
    )


def generate_pdf_report(report_payload: Dict[str, object]) -> bytes:
    lines = [
        "VERIDEX Threat Report",
        "",
        f"Result: {report_payload.get('prediction', report_payload.get('result', 'UNKNOWN'))}",
        f"Confidence: {report_payload.get('confidence', 0)}",
        f"Summary: {generate_threat_report(report_payload)}",
    ]
    # Minimal bytes payload to keep existing API contract usable even without PDF deps.
    # This remains a text-based fallback with application/pdf media type in API layer.
    return ("\n".join(lines)).encode("utf-8")


if __name__ == "__main__":
    generate_detection_report()
