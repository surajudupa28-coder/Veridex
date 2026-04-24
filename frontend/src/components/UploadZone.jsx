import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadMedia } from "../services/api";

const ACCEPTED_FILE_TYPES = "image/*,video/*,audio/*";

function resolveMediaType(file) {
  if (!file?.type) {
    return "unknown";
  }
  if (file.type.startsWith("image/")) {
    return "image";
  }
  if (file.type.startsWith("video/")) {
    return "video";
  }
  if (file.type.startsWith("audio/")) {
    return "audio";
  }
  return "unknown";
}

function UploadZone({ onUploadStateChange }) {
  const inputRef = useRef(null);
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");

  const previewUrl = useMemo(() => {
    if (!selectedFile || !selectedFile.type.startsWith("image/")) {
      return "";
    }
    return URL.createObjectURL(selectedFile);
  }, [selectedFile]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const setState = (state) => {
    if (typeof onUploadStateChange === "function") {
      onUploadStateChange(state);
    }
  };

  const handleSelectedFile = (file) => {
    if (!file) {
      return;
    }
    setSelectedFile(file);
    setErrorMessage("");
    setProgress(0);
    setState({ status: "idle", error: "", result: null, file });
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    handleSelectedFile(event.dataTransfer.files?.[0] ?? null);
  };

  const handleUpload = async () => {
    if (!selectedFile || isUploading) {
      if (!selectedFile) {
        const message = "Please select a media file before uploading.";
        setErrorMessage(message);
        setState({ status: "error", error: message, result: null, file: null });
      }
      return;
    }

    try {
      setIsUploading(true);
      setErrorMessage("");
      setState({ status: "loading", error: "", result: null, file: selectedFile });

      const mediaType = resolveMediaType(selectedFile);
      const response = await uploadMedia(selectedFile, mediaType, (nextProgress) => {
        setProgress(nextProgress);
      });
      const jobId = response?.job_id ?? response?.jobId ?? "";

      setProgress(100);
      setState({ status: "success", error: "", result: response, file: selectedFile });
      navigate("/analysis", {
        state: {
          jobId,
          fileName: selectedFile.name,
          mediaType,
          previewUrl,
          analysis: response?.report ?? null,
          uploadedAt: new Date().toISOString(),
          status: "success",
        },
      });
    } catch (error) {
      const message = error?.response?.data?.detail || error?.message || "Upload failed. Please try again.";
      setErrorMessage(message);
      setState({ status: "error", error: message, result: null, file: selectedFile });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <section className="w-full rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition ${
          isDragging
            ? "border-indigo-500 bg-indigo-50"
            : "border-slate-300 bg-slate-50 hover:border-indigo-400 hover:bg-indigo-50"
        }`}
      >
        <p className="text-lg font-semibold text-slate-800">Drag and drop media here</p>
        <p className="mt-1 text-sm text-slate-500">or click to choose a file</p>
        <p className="mt-3 text-xs text-slate-500">Supported: images, videos, and audio</p>
      </div>

      <input ref={inputRef} type="file" accept={ACCEPTED_FILE_TYPES} className="hidden" onChange={(event) => handleSelectedFile(event.target.files?.[0] ?? null)} />

      <div className="mt-4 space-y-3">
        <p className="truncate text-sm text-slate-700">{selectedFile ? `Selected: ${selectedFile.name}` : "No file selected"}</p>

        {previewUrl ? <img src={previewUrl} alt="Upload preview" className="max-h-52 w-full rounded-lg object-contain" /> : null}

        {isUploading ? (
          <div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
              <div className="h-full rounded-full bg-indigo-600 transition-all" style={{ width: `${progress}%` }} />
            </div>
            <p className="mt-1 text-xs text-slate-500">Uploading... {progress}%</p>
          </div>
        ) : null}

        {errorMessage ? <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700">{errorMessage}</p> : null}

        <div className="flex justify-end">
          <button
            type="button"
            onClick={handleUpload}
            disabled={isUploading}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {isUploading ? "Uploading..." : "Upload Media"}
          </button>
        </div>
      </div>
    </section>
  );
}

export default UploadZone;
