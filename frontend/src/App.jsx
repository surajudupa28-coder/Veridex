import { Navigate, Route, Routes } from "react-router-dom";
import NavBar from "./components/NavBar";
import HistoryView from "./views/HistoryView";
import UploadView from "./views/UploadView";
import AnalysisView from "./views/AnalysisView";

function App() {
  return (
    <div
      style={{
        "--green": "#00ff88",
        "--bg": "#080a0d",
        "--bg2": "#0e1015",
        "--border": "#1c2030",
        "--text": "#d4dae8",
        minHeight: "100vh",
        backgroundColor: "var(--bg)",
        color: "var(--text)",
        fontFamily: 'Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      }}
    >
      <NavBar />
      <main style={{ maxWidth: "1100px", margin: "0 auto", padding: "24px 18px" }}>
        <Routes>
          <Route path="/" element={<UploadView />} />
          <Route path="/analysis" element={<AnalysisView />} />
          <Route path="/history" element={<HistoryView />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <style>{`
        @keyframes veridexPulse {
          0% { box-shadow: 0 0 0 0 rgba(0,255,136,0.7); }
          70% { box-shadow: 0 0 0 12px rgba(0,255,136,0); }
          100% { box-shadow: 0 0 0 0 rgba(0,255,136,0); }
        }
      `}</style>
    </div>
  );
}

export default App;
