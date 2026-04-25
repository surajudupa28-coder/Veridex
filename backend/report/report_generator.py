from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import anthropic  # type: ignore
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore


def _iso_utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _timestamp_id() -> str:
    # Example: 20260425T061523Z
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_analysis_core(analysis: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
    result = str(analysis.get("result") or "").upper().strip()
    confidence = _to_float(analysis.get("confidence"), 0.0)
    component_scores = analysis.get("component_scores") or {}
    if not isinstance(component_scores, dict):
        component_scores = {}
    return result, confidence, component_scores


def _get_flags(analysis: Dict[str, Any]) -> List[str]:
    flags = analysis.get("flags") or analysis.get("indicators") or []
    if isinstance(flags, str):
        flags = [flags]
    if not isinstance(flags, list):
        return []
    out: List[str] = []
    for f in flags:
        if f is None:
            continue
        s = str(f).strip()
        if s:
            out.append(s)
    return out


def _classification(result: str, confidence: float) -> str:
    # Template logic from spec
    if result == "FAKE" and confidence > 70:
        return "THREAT"
    if result == "FAKE":
        return "SUSPICIOUS"
    return "CLEAR"


def _confidence_rating(confidence: float) -> str:
    if confidence > 75:
        return "HIGH"
    if confidence >= 50:
        return "MEDIUM"
    return "LOW"


def _readable_module_name(key: str) -> str:
    # Convert "neural_classifier" -> "Neural Classifier"
    key = (key or "").strip()
    if not key:
        return "Unknown Module"
    return " ".join(part.capitalize() for part in key.replace("-", "_").split("_") if part)


def _risk_interpretation(score: float, module_name: str) -> str:
    # Score expected 0..1; tolerate 0..100.
    s = score
    if s > 1.0:
        s = s / 100.0

    if s >= 0.75:
        return f"HIGH RISK — {module_name} strongly indicates synthetic or manipulated content."
    if s >= 0.5:
        return f"MEDIUM RISK — {module_name} shows signals consistent with possible manipulation."
    return f"LOW RISK — {module_name} detected limited evidence of manipulation."


def _component_findings(component_scores: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for k, v in component_scores.items():
        module = _readable_module_name(str(k))
        score = _to_float(v, 0.0)
        findings.append(
            {
                "module": module,
                "score": score,
                "interpretation": _risk_interpretation(score, module),
            }
        )
    # Keep stable ordering: highest scores first
    findings.sort(key=lambda d: _to_float(d.get("score"), 0.0), reverse=True)
    return findings


def _get_neural_score(component_scores: Dict[str, Any]) -> float:
    # Best-effort: common keys for neural classifier output
    candidates = [
        "neural",
        "neural_score",
        "neural_classifier",
        "neural_classifier_score",
        "nn",
        "ml",
        "deepfake",
        "deepfake_score",
        "classifier",
    ]
    for k in candidates:
        if k in component_scores:
            return _to_float(component_scores.get(k), 0.0)
    # Fallback: use max component score if any
    best = 0.0
    for v in component_scores.values():
        best = max(best, _to_float(v, 0.0))
    return best


def _ioc_sentences(flags: List[str]) -> List[str]:
    out: List[str] = []
    for f in flags:
        cleaned = f.strip().rstrip(".")
        if not cleaned:
            continue
        # Make it read like a sentence.
        if cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        out.append(f"{cleaned}.")
    return out


def _graph_sentence(graph: Optional[Dict[str, Any]]) -> str:
    if not graph or not isinstance(graph, dict):
        return "No network dissemination data was available for this submission."
    stats = graph.get("stats") if isinstance(graph.get("stats"), dict) else {}
    if not stats:
        return "No network dissemination data was available for this submission."

    nodes = stats.get("nodes") or stats.get("node_count")
    edges = stats.get("edges") or stats.get("edge_count")
    communities = stats.get("communities") or stats.get("community_count")
    depth = stats.get("depth") or stats.get("max_depth")
    sources = stats.get("sources") or stats.get("source_count")

    parts: List[str] = []
    if nodes is not None:
        parts.append(f"{int(_to_float(nodes, 0))} accounts/nodes")
    if edges is not None:
        parts.append(f"{int(_to_float(edges, 0))} connections/edges")
    if communities is not None:
        parts.append(f"{int(_to_float(communities, 0))} clusters")
    if depth is not None:
        parts.append(f"depth {int(_to_float(depth, 0))}")
    if sources is not None:
        parts.append(f"{int(_to_float(sources, 0))} apparent source(s)")

    if not parts:
        return "No network dissemination data was available for this submission."
    return "Dissemination signals suggest propagation across " + ", ".join(parts) + "."


def _dissemination_paragraph(graph: Optional[Dict[str, Any]]) -> str:
    if not graph or not isinstance(graph, dict):
        return "N/A — no network data"
    stats = graph.get("stats") if isinstance(graph.get("stats"), dict) else None
    if not stats:
        return "N/A — no network data"

    # Provide a readable paragraph and include raw-ish stats keys if present.
    sentence = _graph_sentence(graph)
    # Add light contextual phrasing without assuming too much.
    extra: List[str] = []
    viral = stats.get("viral_score") or stats.get("virality")
    if viral is not None:
        extra.append(f"Estimated virality score: {_to_float(viral, 0.0):.2f}.")
    timeframe = stats.get("timeframe") or stats.get("window")
    if timeframe:
        extra.append(f"Observation window: {str(timeframe)}.")
    return " ".join([sentence] + extra).strip()


def _attribution_hints(analysis: Dict[str, Any]) -> str:
    metadata = analysis.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    country_hint = metadata.get("country_hint") or metadata.get("country") or metadata.get("geo_hint")
    language = metadata.get("language") or analysis.get("language")

    bits: List[str] = []
    if country_hint:
        bits.append(f"Metadata suggests a possible geographic association with {str(country_hint)}")
    if language:
        bits.append(f"with language signals consistent with {str(language)}")

    if not bits:
        return "Attribution is inconclusive based on available metadata."

    hint = " ".join(bits)
    if not hint.endswith("."):
        hint += "."
    return (
        f"{hint} This is speculative and should be validated using source verification, platform logs, and OSINT."
    )


def _recommended_actions(classification: str) -> List[str]:
    if classification == "THREAT":
        return [
            "Immediately quarantine this content from official channels",
            "File incident report with CERT-In (cert-in.org.in)",
            "Issue internal advisory to command units",
            "Conduct reverse image/video search to find origin",
            "Initiate counter-narrative preparation",
        ]
    if classification == "SUSPICIOUS":
        return [
            "Do not redistribute pending further analysis",
            "Request secondary manual review",
            "Monitor for increased circulation",
        ]
    return [
        "No immediate action required",
        "Archive for future reference",
    ]


def _template_executive_summary(
    classification: str,
    flags: List[str],
    neural_score: float,
    graph: Optional[Dict[str, Any]],
) -> str:
    # Classification level phrase for the requested template line.
    classification_level = {"THREAT": "high", "SUSPICIOUS": "moderate", "CLEAR": "low"}.get(
        classification, "unknown"
    )
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    graph_sentence = _graph_sentence(graph)

    # Normalize neural_score to probability (0..1) if needed.
    p = neural_score
    if p > 1.0:
        p = p / 100.0
    p = max(0.0, min(1.0, p))

    return (
        f"Analysis of submitted media on {date} indicates a {classification_level} probability of synthetic manipulation. "
        f"The ensemble detection system flagged {len(flags)} indicators, with the neural classifier reporting {p * 100:.0f}% fake probability. "
        f"{graph_sentence}"
    )


def _llm_executive_summary(report: Dict[str, Any]) -> Optional[str]:
    if anthropic is None:
        return None
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        prompt_payload = {
            "classification": report.get("classification"),
            "confidence_rating": report.get("confidence_rating"),
            "confidence": report.get("analysis_confidence"),
            "flags": report.get("indicators_of_compromise", []),
            "detection_findings": report.get("detection_findings", []),
            "dissemination_analysis": report.get("dissemination_analysis"),
            "attribution_hints": report.get("attribution_hints"),
        }
        user_prompt = (
            "Write ONLY a concise executive_summary (2–3 sentences) for the following analysis report. "
            "Be neutral, operational, and avoid speculation beyond the provided fields.\n\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False)}"
        )

        msg = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=160,
            temperature=0.2,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = getattr(msg, "content", None)
        # SDK content is commonly a list of blocks; tolerate string too.
        if isinstance(text, str):
            out = text.strip()
            return out or None
        if isinstance(text, list) and text:
            # Look for .text blocks
            parts: List[str] = []
            for block in text:
                t = getattr(block, "text", None)
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            out = " ".join(parts).strip()
            return out or None
    except Exception:
        return None

    return None


def generate_report(analysis: dict, graph: dict | None = None) -> dict:
    result, confidence, component_scores = _get_analysis_core(analysis if isinstance(analysis, dict) else {})
    flags = _get_flags(analysis if isinstance(analysis, dict) else {})

    classification = _classification(result, confidence)
    confidence_rating = _confidence_rating(confidence)
    findings = _component_findings(component_scores)
    neural_score = _get_neural_score(component_scores)

    report: Dict[str, Any] = {
        "report_id": _timestamp_id(),
        "generated_at": _iso_utc_now(),
        "classification": classification,
        "executive_summary": _template_executive_summary(classification, flags, neural_score, graph),
        "detection_findings": findings,
        "indicators_of_compromise": _ioc_sentences(flags),
        "dissemination_analysis": _dissemination_paragraph(graph),
        "attribution_hints": _attribution_hints(analysis if isinstance(analysis, dict) else {}),
        "recommended_actions": _recommended_actions(classification),
        "confidence_rating": confidence_rating,
        # Helpful internal fields for LLM prompt/debug; not part of required list but harmless.
        "analysis_result": result,
        "analysis_confidence": confidence,
    }

    # Optional LLM override for executive_summary
    llm_summary = _llm_executive_summary(report)
    if llm_summary:
        report["executive_summary"] = llm_summary

    return report


def get_report_markdown(report: dict) -> str:
    r: Dict[str, Any] = report if isinstance(report, dict) else {}

    lines: List[str] = []
    lines.append(f"## Veridex Analysis Report — `{r.get('report_id', '')}`")
    lines.append("")
    lines.append(f"- **Generated at**: {r.get('generated_at', '')}")
    lines.append(f"- **Classification**: **{r.get('classification', '')}**")
    lines.append(f"- **Confidence rating**: **{r.get('confidence_rating', '')}**")
    lines.append("")

    lines.append("### Executive summary")
    lines.append(str(r.get("executive_summary", "")).strip() or "N/A")
    lines.append("")

    lines.append("### Detection findings")
    findings = r.get("detection_findings") or []
    if isinstance(findings, list) and findings:
        for f in findings:
            if not isinstance(f, dict):
                continue
            module = str(f.get("module", "Unknown")).strip()
            score = f.get("score", None)
            interp = str(f.get("interpretation", "")).strip()
            score_s = ""
            if score is not None:
                score_s = f"{_to_float(score, 0.0):.3f}"
            lines.append(f"- **{module}**" + (f" (score: `{score_s}`)" if score_s else ""))
            if interp:
                lines.append(f"  - {interp}")
    else:
        lines.append("N/A")
    lines.append("")

    lines.append("### Indicators of compromise")
    iocs = r.get("indicators_of_compromise") or []
    if isinstance(iocs, list) and iocs:
        for i in iocs:
            s = str(i).strip()
            if s:
                lines.append(f"- {s}")
    else:
        lines.append("N/A")
    lines.append("")

    lines.append("### Dissemination analysis")
    lines.append(str(r.get("dissemination_analysis", "")).strip() or "N/A")
    lines.append("")

    lines.append("### Attribution hints")
    lines.append(str(r.get("attribution_hints", "")).strip() or "N/A")
    lines.append("")

    lines.append("### Recommended actions")
    actions = r.get("recommended_actions") or []
    if isinstance(actions, list) and actions:
        for a in actions:
            s = str(a).strip()
            if s:
                lines.append(f"- {s}")
    else:
        lines.append("N/A")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = ["generate_report", "get_report_markdown"]

