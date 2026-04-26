import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

try:
    from backend.config import (
        COMBINED_DATASET_PATH,
        FRAMES_PER_VIDEO,
        IMAGE_SIZE,
        VIDEO_FORMATS,
        VIDEOS_FAKE_PATH,
        VIDEOS_REAL_PATH,
        ensure_dirs,
    )
except ImportError:
    from config import (  # type: ignore
        COMBINED_DATASET_PATH,
        FRAMES_PER_VIDEO,
        IMAGE_SIZE,
        VIDEO_FORMATS,
        VIDEOS_FAKE_PATH,
        VIDEOS_REAL_PATH,
        ensure_dirs,
    )


def _list_videos(path: str) -> List[Path]:
    root = Path(path)
    if not root.exists():
        return []
    ext_set = {e.lower() for e in VIDEO_FORMATS}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set])


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, IMAGE_SIZE)
    normalized = frame.astype(np.float32) / 255.0
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _extract_keyframes(video_path: Path, output_dir: Path, label: str) -> List[Dict[str, str]]:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, max(total_frames - 1, 0), FRAMES_PER_VIDEO, dtype=int).tolist()
    mappings = []
    saved = 0

    for i, frame_idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        processed = _preprocess_frame(frame)
        out_file = output_dir / f"{video_path.stem}_frame_{i:02d}.jpg"
        cv2.imwrite(str(out_file), cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
        mappings.append(
            {
                "source_video": str(video_path),
                "label": label,
                "frame_path": str(out_file),
                "frame_index": int(frame_idx),
            }
        )
        saved += 1

    cap.release()
    if saved == 0:
        return []
    return mappings


def extract_frames() -> None:
    ensure_dirs()
    output_dir = Path(COMBINED_DATASET_PATH) / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    mappings: List[Dict[str, str]] = []
    counts = {"videos": 0, "frames": 0}

    for label, video_dir in (("real", VIDEOS_REAL_PATH), ("fake", VIDEOS_FAKE_PATH)):
        for video_path in _list_videos(video_dir):
            rows = _extract_keyframes(video_path, output_dir, label)
            if rows:
                mappings.extend(rows)
                counts["videos"] += 1
                counts["frames"] += len(rows)

    metadata_path = Path(COMBINED_DATASET_PATH) / "frames_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(mappings, f, indent=2)

    avg = (counts["frames"] / counts["videos"]) if counts["videos"] else 0.0
    print(f"Total frames extracted: {counts['frames']}")
    print(f"Total videos processed: {counts['videos']}")
    print(f"Average frames per video: {avg:.2f}")
    print(f"Saved mapping file: {metadata_path}")


if __name__ == "__main__":
    extract_frames()
