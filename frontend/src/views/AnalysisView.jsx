import { useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { getJobReport, getJobStatus } from "../services/api";
import ResultsPanel from "../components/ResultsPanel";
import ReportPanel from "../components/ReportPanel";

const moduleRows = ["Neural Classifier", "GAN Detector", "Audio Sync", "Metadata Forensics"];

/* ReportPlaceholder replaced by ReportPanel + ResultsPanel */

function AnalysisView({ fetchGraph }) {
  const location = useLocation();
  const navigate = useNavigate();
  const activeJob = location.state ?? null;
  const [statusData, setStatusData] = useState(null);
  const [reportData, setReportData] = useState(activeJob?.analysis ?? null);
  const [errorMessage, setErrorMessage] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const pollRef = useRef(null);
  const timerRef = useRef(null);
  const jobId = activeJob?.jobId ?? "";

  const currentStatus = statusData?.status ?? "queued";
  const liveProgressText = statusData?.progress ?? "Initializing analysis pipeline";
  const isCompleted = currentStatus === "completed";
  const isProcessing = currentStatus === "queued" || currentStatus === "processing";

  useEffect(() => {
    if (!jobId) {
      return undefined;
    }

    let isAlive = true;

    const clearPollers = () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };

    const fetchReport = async () => {
      try {
        const reportJson = await getJobReport(jobId);
        if (isAlive) {
          setReportData(reportJson?.report ?? null);
          if (fetchGraph) {                 // ✅ ADD THIS BLOCK
            fetchGraph(jobId);
          }
        }
        
      } catch (error) {
        if (isAlive) {
          setErrorMessage(error.message || "Failed to fetch report data.");
        }
      }
    };

    const pollStatus = async () => {
      try {
        const statusJson = await getJobStatus(jobId);
        if (!isAlive) {
          return;
        }

        setStatusData(statusJson);

        if (statusJson.status === "failed") {
          clearPollers();
          setErrorMessage(statusJson.error || "Analysis failed on the server.");
          return;
        }

        if (statusJson.status === "completed") {
          clearPollers();
          await fetchReport();
        }
      } catch (error) {
        clearPollers();
        if (isAlive) {
          setErrorMessage(error.message || "Status polling failed.");
        }
      }
    };

    timerRef.current = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);

    pollStatus();
    pollRef.current = setInterval(pollStatus, 1500);

    return () => {
      isAlive = false;
      clearPollers();
    };
  }, [jobId]);

  const elapsedLabel = useMemo(() => `${elapsedSeconds}s`, [elapsedSeconds]);

  if (!jobId) {
    return (
      <section
        style={{
          border: "1px solid var(--border)",
          backgroundColor: "var(--bg2)",
          borderRadius: "14px",
          padding: "20px",
        }}
      >
        <h2 style={{ margin: 0, fontSize: "20px", color: "var(--green)" }}>No Active Analysis</h2>
        <p style={{ marginTop: "10px", color: "var(--text)" }}>
          Open this page from the upload flow so the selected job context is available.
        </p>
        <button
          type="button"
          onClick={() => navigate("/")}
          style={{
            marginTop: "12px",
            border: "1px solid var(--border)",
            backgroundColor: "transparent",
            color: "var(--text)",
            borderRadius: "10px",
            padding: "8px 12px",
            cursor: "pointer",
          }}
        >
          Back to Upload
        </button>
      </section>
    );
  }

  if (errorMessage) {
    return (
      <section
        style={{
          border: "1px solid #5a1d2d",
          backgroundColor: "rgba(90, 29, 45, 0.15)",
          borderRadius: "14px",
          padding: "20px",
          color: "#ff8aa7",
        }}
      >
        <h2 style={{ margin: 0, fontSize: "20px" }}>Analysis Error</h2>
        <p style={{ marginTop: "10px", color: "var(--text)" }}>{errorMessage}</p>
        <button
          type="button"
          onClick={() => navigate("/")}
          style={{
            marginTop: "12px",
            border: "1px solid var(--border)",
            backgroundColor: "transparent",
            color: "var(--text)",
            borderRadius: "10px",
            padding: "8px 12px",
            cursor: "pointer",
          }}
        >
          Retry
        </button>
      </section>
    );
  }

  if (isCompleted && reportData) {
    return (
      <section style={{ display: "grid", gap: "20px" }}>
        {/* Score cards from the raw analysis result */}
        <ResultsPanel reportData={reportData} />

        {/* LLM threat-intel brief */}
        <ReportPanel
          jobId={jobId}
          confidence={Number(reportData?.confidence ?? 0)}
        />

        <button
          type="button"
          onClick={() => navigate("/")}
          style={{
            alignSelf: "start",
            border: "1px solid var(--border)",
            backgroundColor: "transparent",
            color: "var(--text)",
            borderRadius: "10px",
            padding: "8px 12px",
            cursor: "pointer",
          }}
        >
          Back to Upload
        </button>
      </section>
    );
  }

  if (!isProcessing && !isCompleted) {
    return (
      <section
        style={{
          border: "1px solid #5a1d2d",
          backgroundColor: "rgba(90, 29, 45, 0.15)",
          borderRadius: "14px",
          padding: "20px",
          color: "#ff8aa7",
        }}
      >
        <h2 style={{ margin: 0, fontSize: "20px" }}>Analysis Error</h2>
        <p style={{ marginTop: "10px", color: "var(--text)" }}>Unexpected status returned from API.</p>
        <button
          type="button"
          onClick={() => navigate("/")}
          style={{
            marginTop: "12px",
            border: "1px solid var(--border)",
            backgroundColor: "transparent",
            color: "var(--text)",
            borderRadius: "10px",
            padding: "8px 12px",
            cursor: "pointer",
          }}
        >
          Retry
        </button>
      </section>
    );
  }

  return (
    <section
      style={{
        border: "1px solid var(--border)",
        backgroundColor: "var(--bg2)",
        borderRadius: "14px",
        padding: "20px",
      }}
    >
      <h1 style={{ margin: 0, fontSize: "24px", color: "var(--green)" }}>
        Analyzing: {activeJob?.filename ?? activeJob?.fileName ?? "Unknown file"}
      </h1>
      <p
        style={{
          marginTop: "10px",
          marginBottom: "18px",
          fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
          color: "var(--text)",
        }}
      >
        {liveProgressText}
      </p>

      <div
        style={{
          width: "100%",
          height: "18px",
          borderRadius: "999px",
          overflow: "hidden",
          border: "1px solid var(--border)",
          background: "linear-gradient(90deg, #0a1117, #0f1722)",
          marginBottom: "16px",
        }}
      >
        <div className="veridex-scan-bar" />
      </div>

      <div style={{ display: "grid", gap: "10px", marginBottom: "16px" }}>
        {moduleRows.map((moduleName) => (
          <div
            key={moduleName}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              border: "1px solid var(--border)",
              borderRadius: "10px",
              padding: "10px 12px",
              backgroundColor: "var(--bg)",
            }}
          >
            <span style={{ color: "var(--text)" }}>{moduleName}</span>
            <span
              className={isCompleted ? "" : "veridex-pulse-dot"}
              style={{
                width: "10px",
                height: "10px",
                borderRadius: "50%",
                backgroundColor: "var(--green)",
                boxShadow: isCompleted ? "0 0 8px rgba(0,255,136,0.85)" : "0 0 0 0 rgba(0,255,136,0.7)",
              }}
            />
          </div>
        ))}
      </div>

      <p
        style={{
          margin: 0,
          fontSize: "13px",
          color: "var(--text)",
          fontFamily: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, monospace',
        }}
      >
        Elapsed time: {elapsedLabel}
      </p>

      <style>{`
        @keyframes veridexScan {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(350%); }
        }
        @keyframes veridexPulseDot {
          0% { box-shadow: 0 0 0 0 rgba(0,255,136,0.7); opacity: 1; }
          70% { box-shadow: 0 0 0 10px rgba(0,255,136,0); opacity: 0.6; }
          100% { box-shadow: 0 0 0 0 rgba(0,255,136,0); opacity: 1; }
        }
        .veridex-scan-bar {
          width: 35%;
          height: 100%;
          background: linear-gradient(90deg, rgba(0,255,136,0.05), rgba(0,255,136,0.65), rgba(0,255,136,0.05));
          box-shadow: 0 0 14px rgba(0,255,136,0.6);
          animation: veridexScan 1.8s linear infinite;
        }
        .veridex-pulse-dot {
          animation: veridexPulseDot 1.2s ease-in-out infinite;
        }
      `}</style>
    </section>
  );
}

export default AnalysisView;
