"""
VERIDEX Deepfake Detection API (FastAPI main entry point).

This API allows users to upload images or videos for deepfake detection,
monitor job status, retrieve analytic reports (including threat summaries),
and download final reports as PDF.

See: / for listing all endpoints.
"""

import uuid
import time
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from graph.disinfo_graph import build_disinfo_graph
from report.report_generator import generate_report
from report.html_template import render_report_html

import aiofiles

from detectors.image_detector import analyze_image
from detectors.video_detector import analyze_video
from detectors.audio_detector import analyze_audio
from detectors.metadata_detector import analyze_metadata
from detectors.lip_sync_checker import check_lip_sync
from ensemble import (
    aggregate_image_result,
    aggregate_video_result,
    build_threat_summary,
)

logger = logging.getLogger(__name__)

# --- CONSTANTS ---
DATA_DIR = Path("/data")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 100

# Make data dir if not exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- SIMPLE JOB STORE ---
jobs: Dict[str, Dict[str, Any]] = {}

# --- FASTAPI APP & CORS ---
app = FastAPI(title="VERIDEX Deepfake Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- BACKGROUND ANALYSIS TASK ---
async def run_full_analysis(job_id: str, file_path: str, file_ext: str):
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = "detecting faces"

        result = None

        if file_ext in IMAGE_EXTS:
            img_res = analyze_image(file_path)
            metadata_res = analyze_metadata(file_path)
            agg = aggregate_image_result(img_res, metadata_res)
            agg.setdefault("flags", [])
            agg["lip_sync"] = {
                "sync_score": None,
                "flags": [],
                "method": "skipped_not_video"
            }
            agg["threat_summary"] = build_threat_summary(agg)
            result = agg

        elif file_ext in VIDEO_EXTS:
            jobs[job_id]["progress"] = "analyzing video"
            vid_res = analyze_video(file_path)
            jobs[job_id]["progress"] = "analyzing audio"
            audio_res = analyze_audio(file_path)
            metadata_res = analyze_metadata(file_path)
            agg = aggregate_video_result(vid_res, audio_res, metadata_res)
            agg["audio"] = audio_res
            agg.setdefault("flags", [])
            agg["flags"].extend(audio_res.get("flags", []))
            jobs[job_id]["progress"] = "checking lip sync"
            start = time.time()
            try:
                lip_sync_result = check_lip_sync(file_path)
            except Exception as e:
                lip_sync_result = {
                    "sync_score": 0.5,
                    "flags": ["lip_sync_runtime_error"],
                    "method": "mediapipe+librosa",
                    "error": str(e)
                }

            agg["lip_sync"] = lip_sync_result
            agg["flags"].extend(lip_sync_result.get("flags", []))

            logger.info(f"Lip sync executed for {file_path} in {time.time() - start:.2f}s")
            agg["threat_summary"] = build_threat_summary(agg)
            result = agg

        else:
            raise Exception(f"Unsupported file extension: {file_ext}")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["report"] = result
        jobs[job_id]["threat_report"] = generate_report(result)   
        jobs[job_id]["completed_at"] = time.time()
        jobs[job_id].pop("progress", None)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[PIPELINE ERROR] job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id].pop("progress", None)

# --- ENDPOINTS ---

@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Submit an image or video file for deepfake analysis.

    Returns a job_id to poll for status and results.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # Check file size in a streaming, memory-safe way
    file_size_mb = 0
    temp_id = str(uuid.uuid4())
    outfile_path = DATA_DIR / f"{temp_id}{ext}"

    async with aiofiles.open(outfile_path, 'wb') as out_f:
        while chunk := await file.read(1024 * 1024):
            file_size_mb += len(chunk) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                await out_f.close()
                outfile_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large.")
            await out_f.write(chunk)

    jobs[temp_id] = {
        "job_id": temp_id,
        "status": "queued",
        "filename": file.filename,
        "file_path": str(outfile_path),
        "created_at": time.time(),
        "threat_report": None,   
    }

    background_tasks.add_task(run_full_analysis, temp_id, str(outfile_path), ext)

    return {"job_id": temp_id, "status": "queued", "filename": file.filename}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """
    Get the processing status, progress, and basic metadata of a detection job.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    resp = {
        "job_id": job_id,
        "status": job.get("status"),
        "filename": job.get("filename"),
        "created_at": job.get("created_at"),
    }
    if "progress" in job:
        resp["progress"] = job["progress"]
    if "completed_at" in job:
        resp["completed_at"] = job["completed_at"]
    return resp


@app.get("/report/{job_id}/html")
async def export_html(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(status_code=404, detail="Job not found or not completed")
    report = generate_report(job["report"], job.get("graph"))
    html = render_report_html(report)
    return HTMLResponse(
        content=html,
        headers={
            "Content-Disposition": f'attachment; filename="veridex_report_{job_id[:8]}.html"'
        }
    )

@app.get("/report/{job_id}/pdf")
async def export_pdf(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(status_code=404, detail="Job not found or not completed")
    report = generate_report(job["report"], job.get("graph"))
    html = render_report_html(report)
    try:
        import weasyprint
        pdf_bytes = weasyprint.HTML(string=html).write_pdf()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="veridex_{job_id[:8]}.pdf"'
            }
        )
    except ImportError:
        # weasyprint not available — return HTML as fallback
        return HTMLResponse(
            content=html,
            headers={
                "Content-Disposition": f'attachment; filename="veridex_report_{job_id[:8]}.html"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.get("/report/{job_id}")
def get_report(job_id: str):
    """
    Retrieve the full analytic report for a completed job.

    Returns 202 if the job is not complete.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] not in ("completed", "failed"):
        return JSONResponse(
            status_code=202,
            content={"message": "Analysis in progress.", "status": job["status"]}
        )
    if "report" not in job:
        return JSONResponse(
            status_code=500,
            content={"message": "Report unavailable despite completion.", "status": job["status"]}
        )
    return {
    "report": job["report"],
    "threat_report": job.get("threat_report")
    }


@app.get("/health")
def health_check():
    """
    Basic healthcheck for the VERIDEX API service.
    """
    return {
        "status": "ok",
        "jobs_in_memory": len(jobs),
        "version": "1.0.0",
    }


@app.get("/")
def root():
    """
    Service info and endpoint listing.
    """
    return {
        "service": "VERIDEX API",
        "status": "running",
        "endpoints": [route.path for route in app.routes],
    }

#from graph.disinfo_graph import build_disinfo_graph

@app.get("/graph/{job_id}")
async def get_disinfo_graph(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not complete")
    report = job["report"]
    # Only generate graph if content is likely fake
    if report["result"] == "REAL" and report["confidence"] < 40:
        return {"graph": None, "message": "Content appears authentic, no spread analysis needed"}
    media_hash = job_id  # use job_id as deterministic seed
    graph_data = build_disinfo_graph(
        media_hash=media_hash,
        confidence=report["confidence"],
        flags=report.get("flags", []),
        country_hint=report.get("country_hint", "unknown")
    )
    return {"graph": graph_data}