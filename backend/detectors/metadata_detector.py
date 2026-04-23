from __future__ import annotations

import datetime
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import exifread
from PIL import Image

AI_SOFTWARE_SIGNATURES: List[str] = [
    "stable diffusion",
    "midjourney",
    "dall-e",
    "runway",
    "pika",
    "sora",
    "kling",
    "gen-2",
    "adobe firefly",
    "nightcafe",
    "artbreeder",
    "deepfacelab",
    "faceswap",
    "roop",
    "reface",
    "facefusion",
]

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def _extract_ai_signature(value: str) -> Optional[str]:
    lowered = value.lower()
    for signature in AI_SOFTWARE_SIGNATURES:
        if signature in lowered:
            return signature
    return None


def _ratio_to_seconds(value: Any) -> float:
    if hasattr(value, "num") and hasattr(value, "den"):
        den = float(value.den) if float(value.den) != 0 else 1.0
        return float(value.num) / den
    text = str(value)
    if "/" in text:
        left, right = text.split("/", 1)
        right_value = float(right) if float(right) != 0 else 1.0
        return float(left) / right_value
    return float(text)


def _parse_gps_time(gps_time_value: Any) -> Optional[datetime.time]:
    try:
        values = getattr(gps_time_value, "values", gps_time_value)
        if not isinstance(values, (list, tuple)) or len(values) < 3:
            return None
        hour = int(_ratio_to_seconds(values[0]))
        minute = int(_ratio_to_seconds(values[1]))
        second = int(_ratio_to_seconds(values[2]))
        return datetime.time(hour=hour, minute=minute, second=second)
    except Exception:
        return None


def _parse_image_datetime(date_time_value: Any) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.strptime(str(date_time_value), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def _has_expected_video_magic(file_path: Path, extension: str) -> bool:
    try:
        with open(file_path, "rb") as handle:
            header = handle.read(16)
    except Exception:
        return False

    if extension in {".mp4", ".mov"}:
        if len(header) < 12:
            return False
        _size = struct.unpack(">I", header[:4])[0]
        return header[4:8] == b"ftyp"
    if extension == ".avi":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    if extension == ".mkv":
        return len(header) >= 4 and header[:4] == b"\x1a\x45\xdf\xa3"
    return False


def _run_ffprobe(file_path: Path) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(file_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        parsed = json.loads(proc.stdout)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def analyze_metadata(file_path: str) -> Dict[str, Any]:
    flags: List[str] = []
    details: Dict[str, Any] = {"file_path": file_path}
    score = 0.0

    try:
        path = Path(file_path)
        extension = path.suffix.lower()
        details["extension"] = extension
        details["exists"] = os.path.exists(path)

        if extension in _IMAGE_EXTENSIONS:
            exif_tags: Dict[str, Any] = {}
            try:
                with open(path, "rb") as handle:
                    exif_tags = exifread.process_file(handle, stop_tag="UNDEF", details=False)
            except Exception as exif_error:
                details["exif_error"] = str(exif_error)

            details["exif_tag_count"] = len(exif_tags)
            metadata_present = len(exif_tags) > 0

            if not exif_tags:
                flags.append("missing_exif")

            software_tag = exif_tags.get("Image Software")
            if software_tag:
                software_value = str(software_tag)
                details["image_software"] = software_value
                ai_signature = _extract_ai_signature(software_value)
                if ai_signature:
                    flags.append("ai_software_signature")
                    details["ai_software_match"] = ai_signature

            gps_tags = [name for name in exif_tags if name.startswith("GPS ")]
            details["gps_tags"] = gps_tags
            if not gps_tags:
                flags.append("no_gps_data")
            else:
                details["gps_present"] = True

            image_dt_tag = exif_tags.get("Image DateTime")
            gps_time_tag = exif_tags.get("GPS GPSTimeStamp")
            if image_dt_tag:
                details["image_datetime"] = str(image_dt_tag)
            if gps_time_tag:
                details["gps_timestamp"] = str(gps_time_tag)

            parsed_image_dt = _parse_image_datetime(image_dt_tag) if image_dt_tag else None
            parsed_gps_time = _parse_gps_time(gps_time_tag) if gps_time_tag else None
            if parsed_image_dt and parsed_gps_time:
                image_time_seconds = (
                    parsed_image_dt.hour * 3600 + parsed_image_dt.minute * 60 + parsed_image_dt.second
                )
                gps_time_seconds = parsed_gps_time.hour * 3600 + parsed_gps_time.minute * 60 + parsed_gps_time.second
                delta = abs(image_time_seconds - gps_time_seconds)
                # Account for day wrap-around in GPS time.
                delta = min(delta, 86400 - delta)
                details["timestamp_delta_seconds"] = delta
                if delta > 3600:
                    flags.append("timestamp_inconsistency")

            make_tag = exif_tags.get("Image Make")
            model_tag = exif_tags.get("Image Model")
            if make_tag:
                details["camera_make"] = str(make_tag)
            if model_tag:
                details["camera_model"] = str(model_tag)
            if not make_tag and not model_tag:
                flags.append("no_camera_metadata")

            try:
                with Image.open(path) as image:
                    image_info = image.info or {}
                details["pil_info_keys"] = list(image_info.keys())
                comment_value = str(image_info.get("comment", ""))
                if comment_value:
                    details["image_comment"] = comment_value
                    comment_signature = _extract_ai_signature(comment_value)
                    if comment_signature:
                        flags.append("ai_software_signature")
                        details["ai_comment_match"] = comment_signature
            except Exception as pil_error:
                details["pil_error"] = str(pil_error)

        elif extension in _VIDEO_EXTENSIONS:
            metadata_present = False

            has_magic = _has_expected_video_magic(path, extension)
            details["magic_bytes_valid"] = has_magic
            if not has_magic:
                flags.append("invalid_magic_bytes")

            ffprobe_data = _run_ffprobe(path)
            format_data = ffprobe_data.get("format", {}) if isinstance(ffprobe_data, dict) else {}
            tags = format_data.get("tags", {}) if isinstance(format_data, dict) else {}
            details["ffprobe_format"] = format_data
            details["ffprobe_tags"] = tags
            metadata_present = bool(format_data)

            searchable_values = [
                str(format_data.get("encoder", "")),
                str(tags.get("encoder", "")),
                str(tags.get("comment", "")),
                str(tags.get("software", "")),
            ]
            joined_values = " | ".join(v for v in searchable_values if v)
            if joined_values:
                details["encoder_comment_values"] = joined_values
                ai_signature = _extract_ai_signature(joined_values)
                if ai_signature:
                    flags.append("ai_software_signature")
                    details["ai_software_match"] = ai_signature

            creation_time = str(tags.get("creation_time", "")).strip()
            if creation_time:
                details["creation_time"] = creation_time
                if creation_time.startswith("1970-01-01"):
                    flags.append("epoch_creation_time")
        else:
            return {
                "score": 0.0,
                "flags": ["unsupported_format"],
                "metadata_present": False,
                "details": {"file_path": file_path, "extension": extension},
            }

        if "missing_exif" in flags:
            score += 0.3
        if "ai_software_signature" in flags:
            score += 0.6
        if "no_camera_metadata" in flags:
            score += 0.2
        if "timestamp_inconsistency" in flags:
            score += 0.2

        score = min(score, 1.0)
        details["flags_count"] = len(flags)

        return {
            "score": round(score, 2),
            "flags": list(dict.fromkeys(flags)),
            "metadata_present": metadata_present,
            "details": details,
        }
    except Exception as exc:
        return {
            "score": 0.0,
            "flags": ["metadata_analysis_error"],
            "metadata_present": False,
            "details": {"file_path": file_path, "error": str(exc)},
        }
