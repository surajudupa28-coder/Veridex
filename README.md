# VERIDEX

VERIDEX is a full-stack starter setup with:
- React (Vite) frontend
- FastAPI backend
- Shared `models` folder for ML assets
- Shared `data/uploads` folder for uploaded files
- Docker Compose for local development

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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
