import csv
import argparse
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from backend.config import (
        DATASET_FAKE_PATH,
        DATASET_REAL_PATH,
        IMAGE_FORMATS,
        REPORT_PATH,
        VIDEO_FORMATS,
        VIDEOS_FAKE_PATH,
        VIDEOS_REAL_PATH,
        ensure_dirs,
    )
    from backend.predictor import predict_file
except ImportError:
    from config import (  # type: ignore
        DATASET_FAKE_PATH,
        DATASET_REAL_PATH,
        IMAGE_FORMATS,
        REPORT_PATH,
        VIDEO_FORMATS,
        VIDEOS_FAKE_PATH,
        VIDEOS_REAL_PATH,
        ensure_dirs,
    )
    from predictor import predict_file  # type: ignore


def _iter_files(folder: str, exts: Iterable[str]) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    ext_set = {e.lower() for e in exts}
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in ext_set])


def run_batch_prediction(images_only: bool = False, max_files: int = 0) -> Path:
    ensure_dirs()
    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    sources = [
        (DATASET_REAL_PATH, "real", IMAGE_FORMATS),
        (DATASET_FAKE_PATH, "fake", IMAGE_FORMATS),
    ]
    if not images_only:
        sources.extend(
            [
                (VIDEOS_REAL_PATH, "real", VIDEO_FORMATS),
                (VIDEOS_FAKE_PATH, "fake", VIDEO_FORMATS),
            ]
        )

    total_files = 0
    source_files: List[tuple[str, str, List[Path]]] = []
    for folder, actual_label, exts in sources:
        files = _iter_files(folder, exts)
        if max_files > 0:
            files = files[:max_files]
        total_files += len(files)
        source_files.append((actual_label, folder, files))

    print(f"Starting batch prediction for {total_files} files...")
    processed = 0
    for folder, actual_label, exts in sources:
        files = _iter_files(folder, exts)
        if max_files > 0:
            files = files[:max_files]
        for file_path in files:
            try:
                pred = predict_file(str(file_path))
                predicted_label = pred["prediction"].lower()
                confidence = float(pred["confidence"])
                rows.append(
                    {
                        "file_path": str(file_path),
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "is_correct": str(actual_label == predicted_label),
                    }
                )
            except Exception as exc:
                errors.append({"file_path": str(file_path), "error": str(exc)})
            finally:
                processed += 1
                if processed % 25 == 0 or processed == total_files:
                    print(f"Progress: {processed}/{total_files} files processed")

    output_csv = Path(REPORT_PATH) / "predictions_master.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "actual_label",
                "predicted_label",
                "confidence",
                "is_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if errors:
        error_csv = Path(REPORT_PATH) / "prediction_errors.csv"
        with error_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "error"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"Skipped {len(errors)} corrupted/invalid files. See {error_csv}")

    print(f"Saved master predictions to {output_csv}")
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predictor for real/fake datasets.")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Process only image folders and skip videos for faster runs.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional per-source cap for faster dry-runs (0 = no cap).",
    )
    args = parser.parse_args()
    run_batch_prediction(images_only=args.images_only, max_files=args.max_files)
