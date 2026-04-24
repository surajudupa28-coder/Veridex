import { useState } from "react";
import StatusBadge from "../components/StatusBadge";
import UploadZone from "../components/UploadZone";

function UploadView() {
  const [uploadState, setUploadState] = useState({
    status: "idle",
    error: "",
    result: null,
    file: null,
  });

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center">
      <section className="mb-6 rounded-2xl bg-white p-6 shadow-sm">
        <h1 className="text-3xl font-bold text-slate-900">Upload Media</h1>
        <p className="mt-2 text-sm text-slate-600">Upload image, video, or audio content for automated analysis.</p>
      </section>

      <UploadZone onUploadStateChange={setUploadState} />

      <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-700">Upload Status</h2>
          <StatusBadge
            status={
              uploadState.status === "error"
                ? "failed"
                : uploadState.status === "success"
                  ? "success"
                  : "processing"
            }
          />
        </div>
        <p className="mt-3 text-sm text-slate-600">
          {uploadState.error
            ? uploadState.error
            : uploadState.status === "success"
              ? "Upload complete. Redirecting to analysis."
              : "Waiting for file upload."}
        </p>
      </section>
    </div>
  );
}

export default UploadView;
