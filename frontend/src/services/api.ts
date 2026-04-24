import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function uploadMedia(
  file: File,
  mediaType: string,
  onProgress?: (progress: number) => void
) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("mediaType", mediaType);

  const response = await axios.post(`${API_BASE_URL}/analyze`, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
    onUploadProgress: (progressEvent) => {
      if (!onProgress || !progressEvent.total) {
        return;
      }
      const value = Math.round((progressEvent.loaded * 100) / progressEvent.total);
      onProgress(value);
    },
  });

  return response.data;
}

export async function getJobStatus(jobId: string) {
  const response = await axios.get(`${API_BASE_URL}/status/${jobId}`);
  return response.data;
}

export async function getJobReport(jobId: string) {
  const response = await axios.get(`${API_BASE_URL}/report/${jobId}`);
  return response.data;
}
