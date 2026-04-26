import { useEffect, useState, useCallback } from "react";

/* ─── helpers ─────────────────────────────────────────────── */
function formatLocalDatetime(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function formatDateOnly(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString();
  } catch {
    return iso;
  }
}

/* ─── sub-components ──────────────────────────────────────── */

const SECTION_LABEL = {
  fontFamily: "monospace",
  fontSize: "9px",
  letterSpacing: "0.15em",
  textTransform: "uppercase",
  color: "#3d4460",
  display: "block",
  marginBottom: "6px",
};

const SECTION_WRAP = { marginBottom: "20px" };

/* Classification banner colours */
const CLASS_STYLES = {
  THREAT: { bg: "#ff4757", color: "#fff", border: "#ff4757" },
  SUSPICIOUS: { bg: "#ffb347", color: "#1a1009", border: "#ffb347" },
  CLEAR: { bg: "#00ff88", color: "#051a0e", border: "#00ff88" },
};

/* Confidence badge colours */
const RATING_STYLES = {
  HIGH: { bg: "#00ff8820", border: "#00ff88", color: "#00ff88" },
  MEDIUM: { bg: "#ffb34720", border: "#ffb347", color: "#ffb347" },
  LOW: { bg: "#ff475720", border: "#ff4757", color: "#ff4757" },
};

/* ─── skeleton ────────────────────────────────────────────── */
function Skeleton() {
  return (
    <>
      <style>{`
        @keyframes rpSkeletonPulse {
          0%,100% { opacity: 0.4; }
          50%      { opacity: 0.8; }
        }
        .rp-skel { background:#1a1e28; border-radius:6px; height:14px; animation:rpSkeletonPulse 1.4s ease-in-out infinite; }
      `}</style>
      <div style={{ display: "grid", gap: "14px" }}>
        {[["80%"], ["60%"], ["90%"]].map(([w], i) => (
          <div key={i} className="rp-skel" style={{ width: w }} />
        ))}
      </div>
    </>
  );
}

/* ─── main component ──────────────────────────────────────── */
function ReportPanel({ jobId, confidence }) {
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [copied, setCopied]         = useState(false);

  const doFetch = useCallback(() => {
    if (!jobId) return;
    setLoading(true);
    setError(null);

    fetch(`http://localhost:8000/report/${jobId}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const resolvedThreatReport = data?.threat_report ?? data;
        setReportData(resolvedThreatReport);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || "Fetch failed");
        setLoading(false);
      });
  }, [jobId]);

  useEffect(() => {
    doFetch();
  }, [doFetch]);

  if (!jobId) return null;

  /* ── loading ── */
  if (loading) {
    return (
      <div style={panelWrap}>
        <Skeleton />
      </div>
    );
  }

  /* ── error ── */
  if (error || !reportData) {
    return (
      <div style={panelWrap}>
        <div
          style={{
            border: "1px solid #ffb34760",
            backgroundColor: "#ffb34712",
            borderRadius: "8px",
            padding: "14px 16px",
            color: "#ffb347",
            fontFamily: "monospace",
            fontSize: "13px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "12px",
          }}
        >
          <span>⚠ Report not yet available. Analysis may still be processing.</span>
          <button
            type="button"
            onClick={doFetch}
            style={{
              background: "transparent",
              border: "1px solid #ffb347",
              color: "#ffb347",
              borderRadius: "6px",
              padding: "6px 14px",
              cursor: "pointer",
              fontFamily: "monospace",
              fontSize: "12px",
              whiteSpace: "nowrap",
            }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  /* ── data ── */
  const {
    generated_at,
    classification = "CLEAR",
    executive_summary = "",
    detection_findings = [],
    indicators_of_compromise = [],
    recommended_actions = [],
    confidence_rating = "LOW",
    dissemination_analysis = "",
    attribution_hints = "",
  } = reportData;

  const cls  = CLASS_STYLES[classification]  ?? CLASS_STYLES.CLEAR;
  const rtg  = RATING_STYLES[confidence_rating] ?? RATING_STYLES.LOW;

  /* Badge copy text */
  const badgeText = `Analyzed by VERIDEX v1.0 · ${formatDateOnly(generated_at)} · Job ${String(jobId).slice(0, 8).toUpperCase()} · Confidence: ${confidence}%`;

  const handleCopyBadge = () => {
    navigator.clipboard.writeText(badgeText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  const handlePdf = () => {
    setPdfLoading(true);
    window.open(`http://localhost:8000/report/${jobId}/pdf`);
    setTimeout(() => setPdfLoading(false), 3500);
  };

  return (
    <>
      {/* ─── keyframes ─────────────────────────────── */}
      <style>{`
        @keyframes rpBannerPulse {
          0%,100% { opacity:0.4; }
          50%      { opacity:1;   }
        }
        @keyframes rpSpinner {
          to { transform:rotate(360deg); }
        }
      `}</style>

      <div style={panelWrap}>

        {/* 1 ── Classification Banner */}
        <div
          style={{
            backgroundColor: cls.bg,
            color: cls.color,
            textAlign: "center",
            fontFamily: "monospace",
            fontWeight: "bold",
            fontSize: "13px",
            padding: "8px 16px",
            borderRadius: "6px 6px 0 0",
            marginBottom: "16px",
            position: "relative",
            overflow: "hidden",
          }}
        >
          ⬛ VERIDEX CLASSIFICATION: {classification}
          {/* animated pulse border-bottom */}
          <div
            style={{
              position: "absolute",
              bottom: 0,
              left: 0,
              right: 0,
              height: "2px",
              backgroundColor: cls.border,
              animation: "rpBannerPulse 2s ease-in-out infinite",
            }}
          />
        </div>

        {/* 2 ── Confidence badge + timestamp row */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "8px",
            marginBottom: "20px",
          }}
        >
          <span
            style={{
              background: rtg.bg,
              border: `1px solid ${rtg.border}`,
              color: rtg.color,
              borderRadius: "999px",
              padding: "4px 12px",
              fontFamily: "monospace",
              fontSize: "12px",
              fontWeight: 700,
              letterSpacing: "0.06em",
            }}
          >
            {confidence_rating}
          </span>
          <span
            style={{
              fontFamily: "monospace",
              fontSize: "11px",
              color: "#5a6382",
            }}
          >
            Generated {formatLocalDatetime(generated_at)}
          </span>
        </div>

        {/* 3 ── Executive Summary */}
        <div style={SECTION_WRAP}>
          <span style={SECTION_LABEL}>Executive Summary</span>
          <div
            style={{
              borderLeft: "2px solid #4d9fff",
              paddingLeft: "12px",
              fontSize: "13px",
              lineHeight: 1.65,
              color: "#d4dae8",
            }}
          >
            {executive_summary || <span style={{ color: "#5a6382" }}>No summary provided.</span>}
          </div>
        </div>

        {/* 4 ── Recommended Actions */}
        <div style={SECTION_WRAP}>
          <span style={SECTION_LABEL}>Recommended Actions</span>
          <div style={{ display: "grid", gap: "0" }}>
            {recommended_actions.length > 0 ? (
              recommended_actions.map((action, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: "12px",
                    padding: "10px 0",
                    borderBottom: i < recommended_actions.length - 1 ? "1px solid #1c2030" : "none",
                  }}
                >
                  <span
                    style={{
                      flexShrink: 0,
                      minWidth: "26px",
                      height: "26px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      backgroundColor: "#0e1015",
                      border: "1px solid #252b3d",
                      borderRadius: "4px",
                      fontFamily: "monospace",
                      fontSize: "11px",
                      color: "#00e5ff",
                      fontWeight: 700,
                    }}
                  >
                    {i + 1}
                  </span>
                  <span style={{ fontSize: "13px", color: "#d4dae8", lineHeight: 1.55 }}>
                    {action}
                  </span>
                </div>
              ))
            ) : (
              <p style={{ margin: 0, color: "#5a6382", fontSize: "13px" }}>
                No actions recommended.
              </p>
            )}
          </div>
        </div>

        {/* 5 ── Indicators of Compromise */}
        <div style={SECTION_WRAP}>
          <span style={SECTION_LABEL}>Indicators of Compromise</span>
          {indicators_of_compromise.length > 0 ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
              {indicators_of_compromise.map((ioc, i) => (
                <span
                  key={i}
                  style={{
                    background: "#1a1e28",
                    border: "1px solid #252b3d",
                    color: "#ff8c42",
                    fontSize: "11px",
                    fontFamily: "monospace",
                    borderRadius: "4px",
                    padding: "4px 10px",
                  }}
                >
                  {ioc}
                </span>
              ))}
            </div>
          ) : (
            <p style={{ margin: 0, color: "#5a6382", fontSize: "13px" }}>
              No additional indicators flagged.
            </p>
          )}
        </div>

        {/* 6 ── Attribution Hint */}
        {attribution_hints && (
          <div style={SECTION_WRAP}>
            <p
              style={{
                margin: 0,
                fontFamily: "monospace",
                fontSize: "12px",
                color: "#5a6382",
              }}
            >
              <span style={{ marginRight: "6px" }}>Attribution hint:</span>
              <span style={{ color: "#ffb347" }}>{attribution_hints}</span>
            </p>
          </div>
        )}

        {/* 7 ── Export Buttons */}
        <div
          style={{
            display: "flex",
            gap: "10px",
            marginTop: "16px",
            flexWrap: "wrap",
          }}
        >
          {/* HTML export */}
          <button
            type="button"
            id="rp-export-html"
            onClick={() => window.open(`http://localhost:8000/report/${jobId}/html`)}
            style={{
              background: "transparent",
              border: "1px solid #4d9fff",
              color: "#4d9fff",
              borderRadius: "6px",
              padding: "8px 18px",
              cursor: "pointer",
              fontFamily: "monospace",
              fontSize: "13px",
              letterSpacing: "0.03em",
              transition: "background 0.15s",
            }}
          >
            Download HTML Report
          </button>

          {/* PDF export */}
          <button
            type="button"
            id="rp-export-pdf"
            disabled={pdfLoading}
            onClick={handlePdf}
            style={{
              background: pdfLoading ? "#2a4870" : "#4d9fff",
              border: "none",
              color: "#080a0d",
              borderRadius: "6px",
              padding: "8px 18px",
              cursor: pdfLoading ? "not-allowed" : "pointer",
              fontFamily: "monospace",
              fontSize: "13px",
              fontWeight: 700,
              letterSpacing: "0.03em",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              opacity: pdfLoading ? 0.75 : 1,
              transition: "background 0.15s, opacity 0.15s",
            }}
          >
            {pdfLoading ? (
              <>
                <span
                  style={{
                    width: "13px",
                    height: "13px",
                    border: "2px solid rgba(8,10,13,0.3)",
                    borderTop: "2px solid #080a0d",
                    borderRadius: "50%",
                    display: "inline-block",
                    animation: "rpSpinner 0.7s linear infinite",
                  }}
                />
                Generating PDF…
              </>
            ) : (
              "Download PDF Report"
            )}
          </button>
        </div>

        {/* 8 ── Verification Badge */}
        <div
          id="rp-verification-badge"
          title="Click to copy"
          onClick={handleCopyBadge}
          style={{
            marginTop: "14px",
            fontFamily: "monospace",
            fontSize: "11px",
            border: "1px dashed #3d4460",
            padding: "8px 12px",
            borderRadius: "4px",
            color: copied ? "#00ff88" : "#7a8499",
            cursor: "pointer",
            userSelect: "none",
            transition: "color 0.2s",
            letterSpacing: "0.02em",
          }}
        >
          {copied ? "Copied!" : badgeText}
        </div>

      </div>
    </>
  );
}

/* ─── panel container style ───────────────────────────────── */
const panelWrap = {
  backgroundColor: "#0e1015",
  border: "1px solid #1c2030",
  borderRadius: "8px",
  padding: "1.25rem",
};

export default ReportPanel;
