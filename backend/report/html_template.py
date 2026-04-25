from __future__ import annotations

from html import escape
from typing import Any, Dict, Iterable, List


def _get_str(report: Dict[str, Any], key: str, default: str = "N/A") -> str:
    val = report.get(key, default)
    if val is None:
        return default
    return str(val)


def _classification_color(classification: str) -> str:
    c = (classification or "").upper().strip()
    if c == "THREAT":
        return "#ff4757"
    if c == "SUSPICIOUS":
        return "#ffb347"
    return "#00cc6a"


def _confidence_color(confidence_rating: str) -> str:
    r = (confidence_rating or "").upper().strip()
    # Spec: HIGH=red, MEDIUM=amber, LOW=green
    if r == "HIGH":
        return "#ff4757"
    if r == "MEDIUM":
        return "#ffb347"
    return "#00cc6a"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _fmt_score(v: Any) -> str:
    try:
        if v is None:
            return "N/A"
        f = float(v)
        return f"{f:.3f}"
    except (TypeError, ValueError):
        s = str(v).strip()
        return s or "N/A"


def render_report_html(report: dict) -> str:
    r: Dict[str, Any] = report if isinstance(report, dict) else {}

    classification = _get_str(r, "classification", "N/A").upper()
    class_color = _classification_color(classification)

    report_id = _get_str(r, "report_id", "N/A")
    generated_at = _get_str(r, "generated_at", "N/A")
    executive_summary = _get_str(r, "executive_summary", "N/A")

    confidence_rating = _get_str(r, "confidence_rating", "N/A").upper()
    confidence_color = _confidence_color(confidence_rating)

    dissemination = _get_str(r, "dissemination_analysis", "N/A")
    attribution = _get_str(r, "attribution_hints", "N/A")

    findings = r.get("detection_findings", [])
    findings_list: List[Dict[str, Any]] = []
    if isinstance(findings, list):
        for item in findings:
            if isinstance(item, dict):
                findings_list.append(item)

    iocs = _as_list(r.get("indicators_of_compromise", []))
    ioc_items = [str(x).strip() for x in iocs if str(x).strip()]

    actions = _as_list(r.get("recommended_actions", []))
    action_items = [str(x).strip() for x in actions if str(x).strip()]

    rows_html = []
    for item in findings_list:
        module = str(item.get("module", "N/A")).strip() or "N/A"
        score = _fmt_score(item.get("score", "N/A"))
        interpretation = str(item.get("interpretation", "N/A")).strip() or "N/A"
        rows_html.append(
            "<tr>"
            f"<td>{escape(module)}</td>"
            f"<td class='score'>{escape(score)}</td>"
            f"<td>{escape(interpretation)}</td>"
            "</tr>"
        )
    tbody_inner = "".join(rows_html) if rows_html else '<tr><td colspan="3">N/A</td></tr>'
    findings_table = (
        "<table class='findings'>"
        "<thead><tr><th>MODULE</th><th>SCORE</th><th>INTERPRETATION</th></tr></thead>"
        f"<tbody>{tbody_inner}</tbody>"
        "</table>"
    )

    ioc_html = (
        "<ul class='ioc'>"
        + "".join([f"<li><span class='bullet'>▶</span> {escape(x)}</li>" for x in ioc_items])
        + "</ul>"
        if ioc_items
        else "<div class='muted'>No indicators flagged.</div>"
    )

    actions_html = (
        "<ol class='actions'>"
        + "".join([f"<li><b>{escape(x)}</b></li>" for x in action_items])
        + "</ol>"
        if action_items
        else "<div class='muted'>N/A</div>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VERIDEX Threat Intelligence Report</title>
  <style>
    :root {{
      --bg: #0a0c10;
      --panel: #0d1017;
      --panel2: #0b0e14;
      --text: #e8eef9;
      --muted: #9aa6b2;
      --dim: #6b7280;
      --line: #1c2030;
      --table-line: #333;
      --heading: #4d9fff;
      --accent: {class_color};
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Courier New", Courier, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      line-height: 1.45;
    }}
    .page {{
      max-width: 800px;
      margin: 0 auto;
      padding: 40px;
    }}
    .banner {{
      background: #07090d;
      border: 1px solid var(--line);
      padding: 18px 14px;
      text-align: center;
      letter-spacing: 0.08em;
      font-size: 18px;
      font-weight: 700;
    }}
    .class-strip {{
      margin-top: 10px;
      border: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(0,0,0,0.15), rgba(255,255,255,0.03));
      padding: 10px 12px;
      font-weight: 700;
      color: #0a0c10;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .class-strip .pill {{
      display: inline-block;
      padding: 6px 10px;
      background: var(--accent);
      border: 1px solid rgba(255,255,255,0.12);
    }}
    .meta {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      border: 1px solid var(--line);
      background: var(--panel2);
      padding: 10px 12px;
    }}
    .meta .k {{ color: #cbd5e1; }}

    .section {{
      margin-top: 24px;
      padding-top: 0;
      page-break-inside: avoid;
    }}
    .section-title {{
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.15em;
      color: var(--heading);
      border-bottom: 1px solid var(--line);
      padding-bottom: 8px;
      margin-bottom: 12px;
    }}
    .summary {{
      color: #d5dbe7;
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 14px 14px;
    }}
    .badge {{
      display: inline-block;
      padding: 8px 12px;
      border: 1px solid var(--line);
      background: var(--panel2);
      font-weight: 700;
      letter-spacing: 0.08em;
    }}
    .badge b {{
      color: {confidence_color};
    }}

    table.findings {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--table-line);
      background: #0b0d12;
    }}
    table.findings th, table.findings td {{
      border: 1px solid var(--table-line);
      padding: 10px 10px;
      vertical-align: top;
      font-size: 12px;
    }}
    table.findings th {{
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: #cfe0ff;
      background: #090b10;
    }}
    table.findings tbody tr:nth-child(odd) td {{ background: #111; }}
    table.findings tbody tr:nth-child(even) td {{ background: #0d0d0d; }}
    td.score {{
      white-space: nowrap;
      color: #e6f0ff;
      font-weight: 700;
    }}

    ul.ioc {{
      list-style: none;
      margin: 0;
      padding: 0;
    }}
    ul.ioc li {{
      margin: 8px 0;
      padding: 10px 10px;
      border: 1px solid var(--line);
      background: var(--panel2);
    }}
    .bullet {{
      color: var(--accent);
      margin-right: 10px;
    }}

    ol.actions {{
      margin: 0;
      padding-left: 20px;
    }}
    ol.actions li {{
      margin: 10px 0;
      padding: 10px 10px;
      border: 1px solid var(--line);
      background: var(--panel);
    }}

    .muted {{ color: var(--muted); }}
    .italic {{ font-style: italic; color: #d5dbe7; }}
    .disclaimer {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    hr {{
      border: none;
      border-top: 1px solid var(--line);
      margin: 26px 0 12px;
    }}
    .footer {{
      color: var(--dim);
      font-size: 12px;
    }}

    @media print {{
      body {{
        background: #ffffff;
        color: #000000;
      }}
      .page {{
        padding: 24px;
      }}
      .banner, .meta, .summary, ul.ioc li, ol.actions li, table.findings {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="banner">VERIDEX THREAT INTELLIGENCE REPORT</div>
    <div class="class-strip"><span class="pill">⬛ CLASSIFICATION: {escape(classification or "N/A")}</span></div>
    <div class="meta">
      <div><span class="k">REPORT ID</span>: {escape(report_id)}</div>
      <div><span class="k">GENERATED AT</span>: {escape(generated_at)}</div>
    </div>

    <div class="section">
      <div class="section-title">EXECUTIVE SUMMARY</div>
      <div class="summary">{escape(executive_summary)}</div>
    </div>

    <div class="section">
      <div class="badge">CONFIDENCE RATING: <b>{escape(confidence_rating or "N/A")}</b></div>
    </div>

    <div class="section">
      <div class="section-title">DETECTION FINDINGS</div>
      {findings_table}
    </div>

    <div class="section">
      <div class="section-title">INDICATORS OF COMPROMISE (IOC)</div>
      {ioc_html}
    </div>

    <div class="section">
      <div class="section-title">DISSEMINATION ANALYSIS</div>
      <div class="summary">{escape(dissemination)}</div>
    </div>

    <div class="section">
      <div class="section-title">ATTRIBUTION (SPECULATIVE)</div>
      <div class="summary italic">{escape(attribution)}</div>
      <div class="disclaimer">⚠ Attribution data is speculative and should not be used as sole basis for operational decisions.</div>
    </div>

    <div class="section">
      <div class="section-title">RECOMMENDED ACTIONS</div>
      {actions_html}
    </div>

    <hr />
    <div class="footer">This report was generated automatically by VERIDEX v1.0. For official use only. Do not distribute without authorization.</div>
  </div>
</body>
</html>
"""


__all__ = ["render_report_html"]

