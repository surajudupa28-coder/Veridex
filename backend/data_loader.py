import csv
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance

try:
    from backend.config import (
        COMBINED_DATASET_PATH,
        DATASET_FAKE_PATH,
        DATASET_REAL_PATH,
        IMAGE_FORMATS,
        IMAGE_SIZE,
        TEST_SPLIT,
        VALIDATION_SPLIT,
        VIDEOS_FAKE_PATH,
        VIDEOS_REAL_PATH,
        VIDEO_FORMATS,
        ensure_dirs,
    )
except ImportError:
    from config import (  # type: ignore
        COMBINED_DATASET_PATH,
        DATASET_FAKE_PATH,
        DATASET_REAL_PATH,
        IMAGE_FORMATS,
        IMAGE_SIZE,
        TEST_SPLIT,
        VALIDATION_SPLIT,
        VIDEOS_FAKE_PATH,
        VIDEOS_REAL_PATH,
        VIDEO_FORMATS,
        ensure_dirs,
    )


SPLITS = ("train", "validation", "test")
RNG = random.Random(42)


def _list_valid_files(folder: str, extensions: List[str]) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    valid = []
    ext_set = {e.lower() for e in extensions}
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ext_set:
            valid.append(file_path)
    return sorted(valid)


def _split_indices(total: int) -> Dict[str, List[int]]:
    idx = list(range(total))
    RNG.shuffle(idx)
    val_count = int(total * VALIDATION_SPLIT)
    test_count = int(total * TEST_SPLIT)
    train_count = total - val_count - test_count
    return {
        "train": idx[:train_count],
        "validation": idx[train_count : train_count + val_count],
        "test": idx[train_count + val_count :],
    }


def _augment_image(image: Image.Image) -> List[Image.Image]:
    augmented = []
    angle = RNG.uniform(-15, 15)
    augmented.append(image.rotate(angle))
    brightness = ImageEnhance.Brightness(image).enhance(RNG.uniform(0.8, 1.2))
    augmented.append(brightness)
    contrast = ImageEnhance.Contrast(image).enhance(RNG.uniform(0.8, 1.2))
    augmented.append(contrast)
    if RNG.random() < 0.5:
        augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    arr = np.array(image)
    k = RNG.choice([3, 5])
    blurred = cv2.GaussianBlur(arr, (k, k), 0)
    augmented.append(Image.fromarray(blurred))
    return augmented


def _normalize_and_save(image: Image.Image, out_path: Path) -> Tuple[int, int]:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr_255 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_255).save(out_path, quality=95)
    return image.width, image.height


def _video_dimensions(path: Path) -> Tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return width, height


def _write_metadata(metadata_rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_path",
                "label",
                "source",
                "file_size",
                "dimensions",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)


def prepare_dataset() -> None:
    ensure_dirs()
    combined_root = Path(COMBINED_DATASET_PATH)
    if combined_root.exists():
        shutil.rmtree(combined_root)
    ensure_dirs()

    image_sources = {
        "real": _list_valid_files(DATASET_REAL_PATH, IMAGE_FORMATS),
        "fake": _list_valid_files(DATASET_FAKE_PATH, IMAGE_FORMATS),
    }
    video_sources = {
        "real": _list_valid_files(VIDEOS_REAL_PATH, VIDEO_FORMATS),
        "fake": _list_valid_files(VIDEOS_FAKE_PATH, VIDEO_FORMATS),
    }

    min_images = min(len(image_sources["real"]), len(image_sources["fake"])) if image_sources else 0
    balanced_images = {
        "real": image_sources["real"][:min_images],
        "fake": image_sources["fake"][:min_images],
    }

    metadata_rows: List[Dict[str, str]] = []
    totals = Counter()

    for label, files in balanced_images.items():
        split_map = _split_indices(len(files))
        for split, ids in split_map.items():
            for idx in ids:
                src = files[idx]
                dst = combined_root / split / label / f"{src.stem}.jpg"
                with Image.open(src) as image:
                    width, height = _normalize_and_save(image, dst)
                    metadata_rows.append(
                        {
                            "file_path": str(dst),
                            "label": label,
                            "source": "image",
                            "file_size": str(dst.stat().st_size),
                            "dimensions": f"{width}x{height}",
                            "split": split,
                        }
                    )
                    totals[f"image_{label}"] += 1

                    if split == "train":
                        for aug_idx, aug_image in enumerate(_augment_image(image), start=1):
                            aug_dst = combined_root / split / label / f"{src.stem}_aug_{aug_idx}.jpg"
                            w2, h2 = _normalize_and_save(aug_image, aug_dst)
                            metadata_rows.append(
                                {
                                    "file_path": str(aug_dst),
                                    "label": label,
                                    "source": "image",
                                    "file_size": str(aug_dst.stat().st_size),
                                    "dimensions": f"{w2}x{h2}",
                                    "split": split,
                                }
                            )
                            totals[f"image_{label}"] += 1

    for label, videos in video_sources.items():
        split_map = _split_indices(len(videos))
        for split, ids in split_map.items():
            for idx in ids:
                src = videos[idx]
                width, height = _video_dimensions(src)
                metadata_rows.append(
                    {
                        "file_path": str(src),
                        "label": label,
                        "source": "video",
                        "file_size": str(src.stat().st_size),
                        "dimensions": f"{width}x{height}",
                        "split": split,
                    }
                )
                totals[f"video_{label}"] += 1

    _write_metadata(metadata_rows, combined_root / "metadata.csv")

    print("Data preparation completed.")
    print(f"Total images (balanced + augmented): {totals['image_real'] + totals['image_fake']}")
    print(f"Total videos: {totals['video_real'] + totals['video_fake']}")
    print(
        "Class distribution: "
        f"real={totals['image_real'] + totals['video_real']}, "
        f"fake={totals['image_fake'] + totals['video_fake']}"
    )


if __name__ == "__main__":
    prepare_dataset()
