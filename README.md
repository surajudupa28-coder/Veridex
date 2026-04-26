# VERIDEX

VERIDEX is a full-stack starter setup with:
- React (Vite) frontend
- FastAPI backend
- Shared `models` folder for ML assets
- Shared `data/uploads` folder for uploaded files
- Docker Compose for local development

## Project Overview

VERIDEX is an end-to-end deepfake analysis platform for image and video media. It provides:

- A web UI for upload, live status tracking, and report visualization.
- A FastAPI backend that accepts media, runs asynchronous detector pipelines, and exposes reports.
- Multi-signal ensemble scoring (neural image/video signals + GAN artifact signals + metadata + audio heuristics).
- Threat-intelligence style reporting with optional HTML/PDF export.
- Simulated disinformation spread graph generation for suspicious/high-risk outputs.

The current implementation is optimized for local development and demo/hackathon environments with CPU-first inference defaults.

## Architecture

### High-level flow

1. User uploads media from the frontend (`UploadZone`).
2. Backend stores file under `/data` and returns a `job_id`.
3. Frontend routes to analysis page and polls `/status/{job_id}`.
4. Backend background task runs media-specific detection pipeline.
5. Aggregated result + threat report are stored in in-memory job state.
6. Frontend fetches `/report/{job_id}` once complete.
7. Frontend optionally fetches `/graph/{job_id}` and renders propagation network with D3.
8. User can export report as HTML/PDF via backend endpoints.

### Components

- `frontend/`:
  - React + Vite + Tailwind UI.
  - Upload workflow, status polling, report display, graph visualization.
- `backend/`:
  - FastAPI app (`main.py`).
  - Detector modules (`detectors/`) for image, video, audio, metadata.
  - Ensemble logic (`ensemble.py`) for weighted scoring and summaries.
  - Report generation (`report/`) for JSON + HTML/PDF rendering.
  - Graph simulation (`graph/disinfo_graph.py`) for dissemination analysis.
- Shared volumes:
  - `./data` mounted to backend `/data` for uploaded inputs.
  - `./models` mounted to backend `/app/models` for model/cache persistence.

## Feature Matrix

### Core features

- Upload and analyze image/video files through API and UI.
- Asynchronous job queue behavior (in-memory) with progress stages.
- Detector-level signal extraction:
  - Image neural classifier (EfficientNet-B4).
  - GAN artifact heuristic (DCT high/low frequency energy ratio).
  - Video frame sampling + per-frame image analysis.
  - Audio deep + prosody analysis (Wav2Vec2 embedding path with prosody fallback).
  - Lip-sync consistency check (MediaPipe face landmarks + librosa audio energy correlation).
  - Metadata forensics (EXIF/ffprobe signatures and consistency checks).
- Weighted ensemble decision engine for final `REAL` / `FAKE`.
- Threat summary generation from flags/components.
- Intelligence-style report generation:
  - Classification (`THREAT`, `SUSPICIOUS`, `CLEAR`)
  - IOC list
  - Recommended actions
  - Dissemination and attribution sections
- HTML and PDF report export.
- Propagation graph generation and D3 force-directed graph rendering.
- Health and endpoint discovery routes (`/health`, `/`).

### Frontend features

- Drag-and-drop or click-to-select media upload.
- Upload progress bar.
- File preview for image uploads.
- Analysis page with:
  - Polling-based state updates.
  - Pipeline activity indicator UI.
  - Elapsed timer.
  - Completed report JSON panel.
- Graph visualization panel with:
  - Node hover tooltips.
  - Bot/origin/amplifier visual differentiation.
  - Summary stats cards.
- History view scaffold (currently mock data).

## Detection Pipelines (Detailed)

### 1) Image pipeline

Implemented across `detectors/image_detector.py`, `detectors/metadata_detector.py`, and `ensemble.py`.

1. Load image and normalize to RGB.
2. Detect faces with MTCNN:
   - If no valid face boxes, fallback to full-image analysis and flag `no_face_detected`.
3. Classify each face crop with EfficientNet-B4-based binary head (`real` vs `fake` logits).
4. Average per-face fake probabilities.
5. Run GAN artifact heuristic using DCT energy ratio.
6. Build image detector flags (for example `gan_artifacts_detected`, `high_face_fake_probability`).
7. Run metadata detector for EXIF consistency and signature analysis.
8. Aggregate image + metadata via weighted score:
   - `neural`: 0.55
   - `gan`: 0.20
   - `metadata`: 0.25
9. Threshold at `0.52` to produce final `FAKE` or `REAL`.
10. Build plain-English threat summary.

### 2) Video pipeline (with lip-sync check)

Implemented across `detectors/video_detector.py`, `detectors/audio_detector.py`, `detectors/metadata_detector.py`, and `ensemble.py`.

1. Open video with OpenCV; fail early if stream cannot be decoded.
2. Derive FPS and sample every ~0.5 seconds (approximately 2 FPS).
3. Convert sampled frames to JPEG temp files and run `analyze_image` per sampled frame.
4. Build frame timeline and per-frame confidence/result flags.
5. Compute video-level frame metrics:
   - `total_frames_analyzed`
   - `fake_frames`
   - `fake_ratio`
   - `duration_sec`
6. Run audio extraction with `ffmpeg`:
   - mono, 16 kHz WAV temporary file.
7. Run audio analysis:
   - Prosody features (speaking rate, pitch variance, unnatural pauses).
   - Optional Wav2Vec2 embedding + lightweight classifier when model loading succeeds.
   - Graceful fallback to prosody-only scoring when transformer weights are unavailable.
8. Run lip-sync verification (`detectors/lip_sync_checker.py`):
   - extracts first 5 seconds of frames/audio with `ffmpeg`
   - computes mouth openness per frame using MediaPipe FaceMesh landmarks
   - computes correlation between mouth-motion signal and audio RMS envelope
   - emits `sync_score`, `correlation`, and flags such as `poor_lip_sync`
9. Run metadata detector using ffprobe + magic bytes + tag checks.
10. Aggregate video + audio + metadata via weighted score:
   - `neural`: 0.50
   - `gan`: 0.15
   - `audio`: 0.20
   - `metadata`: 0.15
11. Append audio and lip-sync flags into final output and build threat summary.
12. Threshold at `0.52` for final verdict.

### 3) Metadata forensics pipeline

Implemented in `detectors/metadata_detector.py`.

- Image metadata checks:
  - EXIF presence (`no_exif_data` behavior)
  - AI software signatures in EXIF comments/software tags
  - GPS tag presence / absence
  - timestamp consistency between image datetime and GPS timestamp
  - camera make/model availability
- Video metadata checks:
  - magic byte validation by container type
  - ffprobe format/tag extraction
  - encoder/comment/software AI signature matching
  - epoch-like suspicious creation timestamp
- Produces metadata score (`0..1`) and flags consumed by ensemble aggregation.

### 4) Audio analysis pipeline

Implemented in `detectors/audio_detector.py`.

- Extracts mono 16 kHz audio from video (`ffmpeg`) and truncates long clips.
- Computes prosody signals with librosa:
  - speaking rate from RMS peak structure
  - pitch variance (`yin`)
  - pause irregularity
- Attempts Wav2Vec2 embedding inference (`facebook/wav2vec2-base`) and combines:
  - `0.6 * wav2vec_score + 0.4 * prosody_fake_score`
- Falls back to prosody-only score if transformer/model loading fails.
- Returns robust neutral/error outputs (`score=0.5`) with explicit flags when extraction/analysis fails.

### 5) Lip-sync verification pipeline

Implemented in `detectors/lip_sync_checker.py`.

- Extracts short frame and audio windows (first 5 seconds) for lightweight consistency checks.
- Uses MediaPipe FaceMesh landmarks (`13`, `14`, `10`, `152`) to estimate normalized mouth openness.
- Correlates mouth openness series with audio RMS envelope.
- Emits:
  - `sync_score` (`0..1`)
  - `correlation`
  - `frames_analyzed`
  - `flags` (`poor_lip_sync`, `no_face_for_sync`, `sync_check_failed`)
- Includes guarded fallback output on runtime issues to avoid pipeline crashes.

## Report Generation Pipeline

Implemented in `report/report_generator.py` and `report/html_template.py`.

1. Ingest aggregated analysis output.
2. Determine classification:
   - `THREAT` => `FAKE` with high confidence (>70)
   - `SUSPICIOUS` => `FAKE` otherwise
   - `CLEAR` => non-fake verdict
3. Rate confidence (`HIGH`, `MEDIUM`, `LOW`).
4. Build component findings ordered by descending risk score.
5. Convert detector flags to IOC-style readable statements.
6. Build dissemination section from optional graph stats.
7. Build attribution hints from metadata hints when available.
8. Attach response playbook actions by classification.
9. Render as:
   - JSON (API response)
   - HTML (download endpoint)
   - PDF via WeasyPrint (fallback to HTML if unavailable)

Optional enhancement: if `ANTHROPIC_API_KEY` is configured and the dependency exists, an LLM can generate an alternate concise executive summary.

## Disinformation Graph Pipeline

Implemented in `graph/disinfo_graph.py` and exposed by `/graph/{job_id}`.

- Deterministic seeded graph generation based on `job_id`/media hash.
- Produces:
  - origin node
  - amplifier nodes
  - propagator nodes
  - weighted/time-offset edges
  - derived stats (`total_nodes`, `bot_account_count`, `reach_estimate`, etc.)
- Frontend `DisInfoGraph` renders force simulation using D3:
  - color/radius by node role and bot score
  - drag interactions
  - tooltips + summary cards

Graph generation is currently simulated for UX/demo and should be replaced by live platform ingestion for production.

## API Reference

Base URL (local): `http://localhost:8000`

### `POST /analyze`
- Multipart upload endpoint for supported image/video files.
- Stores file, creates in-memory job, and schedules background processing.
- Returns:
  - `job_id`
  - `status` (`queued`)
  - `filename`

### `GET /status/{job_id}`
- Returns job lifecycle and progress metadata:
  - `queued` / `processing` / `completed` / `failed`
  - optional `progress`
  - timestamps and filename

### `GET /report/{job_id}`
- Returns `202` while processing.
- Returns report payload when complete:
  - `report` (aggregated detector output)
  - `threat_report` (intelligence report object)

### `GET /report/{job_id}/html`
- Generates and returns report as downloadable HTML.

### `GET /report/{job_id}/pdf`
- Generates PDF via WeasyPrint.
- Falls back to downloadable HTML if PDF library unavailable.

### `GET /graph/{job_id}`
- Returns disinformation graph JSON for completed jobs.
- Skips graph for low-risk authentic content path.

### `GET /health`
- Service health check + in-memory job count.

### `GET /`
- Service metadata and discovered route list.

## Frontend App Flow

### Upload page (`/`)
- Accepts `image/*`, `video/*`, `audio/*` (frontend selector).
- Determines media type from MIME and uploads via `uploadMedia`.
- On success, navigates to `/analysis` with job context.

### Analysis page (`/analysis`)
- Polls job status every 1.5s.
- Shows live progress stage and animated module indicators.
- Fetches completed report and triggers graph fetch.

### History page (`/history`)
- Placeholder/mock history list to be replaced by persisted backend job history.

## Model & Training Assets

- Inference model architecture:
  - `timm` EfficientNet-B4 backbone
  - custom binary classifier head
  - CPU execution by default in detector module
- Training utility: `backend/train.py`
  - expects separate real/fake image directories
  - builds combined dataset
  - fine-tunes classifier head
  - saves best checkpoint by validation accuracy

## Testing / Utilities

### CLI image test tool

`backend/test_image.py` provides:
- single file or directory batch processing
- validation and robust error handling
- output modes:
  - human-readable
  - JSON
  - CSV
- optional output-to-file and verbose logging

## Docker & Runtime Configuration

### Docker Compose services

- `backend`:
  - build context: `./backend`
  - exposes `8000`
  - mounts:
    - `./backend:/app`
    - `./data:/data`
    - `./models:/app/models`
  - dev command: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

- `frontend`:
  - build context: `./frontend`
  - exposes `5173`
  - uses `VITE_API_BASE_URL=http://localhost:8000`
  - mounts app code for live development

### Backend container notes

- Python 3.11 slim base image.
- Includes native libs for:
  - OpenCV runtime
  - ffmpeg / libsndfile audio processing
  - WeasyPrint rendering dependencies
  - MIME/file inspection support
- Sets:
  - `HF_HOME=/app/models`
  - `TORCH_HOME=/app/models`
  to keep downloaded model/cache assets on mounted volume.

## Project Structure

```text
VERIDEX/
  backend/
    detectors/
      image_detector.py
      video_detector.py
      audio_detector.py
      metadata_detector.py
    graph/
      disinfo_graph.py
    report/
      report_generator.py
      html_template.py
    main.py
    ensemble.py
    train.py
    test_image.py
    requirements.txt
    Dockerfile
  frontend/
    src/
      components/
      services/
      views/
    package.json
    Dockerfile
  data/
  models/
  docker-compose.yml
  README.md
```

## Requirements

### Backend
- Python 3.11+
- ffmpeg available in runtime
- dependencies listed in `backend/requirements.txt`

### Frontend
- Node.js 20+ recommended
- npm

## Quick Start

### 1) Run with Docker Compose

```bash
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 2) Run without Docker

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

#### Backend

```bash
cd backend
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Compatibility Notes

- Current backend code entrypoint is `main:app` in the `backend` directory (Docker and local run).
- In-memory job storage means status/report data resets on backend restart.

If your local command still uses `uvicorn app.main:app ...`, update it to:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Limitations (Current State)

- No persistent database for jobs/history.
- No authentication/authorization layer.
- Graph pipeline uses seeded simulation, not live social feed ingestion.
- Audio-only uploads are accepted by frontend selector, but backend `analyze` currently supports image/video extensions.
- No distributed task queue (single-process in-memory background tasks).

## Suggested Next Enhancements

- Add persistent job store (SQLite/PostgreSQL + ORM).
- Move background work to task queue (Celery/RQ/Arq) for scalability.
- Implement true upload history API and wire `HistoryView` to backend.
- Add auth and role-based access controls.
- Integrate real dissemination telemetry sources for graph pipeline.
- Add automated unit/integration tests and CI quality gates.
