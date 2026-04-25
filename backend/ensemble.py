"""Aggregation utilities for combining deepfake detector outputs."""

from __future__ import annotations

from typing import Any, Dict, List

IMAGE_WEIGHTS = {"neural": 0.55, "gan": 0.20, "metadata": 0.25}
VIDEO_WEIGHTS = {"neural": 0.50, "gan": 0.15, "audio": 0.20, "metadata": 0.15}
FAKE_THRESHOLD = 0.52


def _combine_flags(*result_dicts: Dict[str, Any]) -> List[str]:
    """Combine and deduplicate detector flags while preserving first-seen order."""
    combined: List[str] = []
    seen = set()
    for result in result_dicts:
        for flag in result.get("flags", []):
            if flag not in seen:
                seen.add(flag)
                combined.append(flag)
    return combined


def aggregate_image_result(image_result: Dict[str, Any], metadata_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate image + metadata detector signals into one final image verdict.

    Args:
        image_result: Output from image detector with raw_scores.face_score and raw_scores.gan_score.
        metadata_result: Output from metadata detector with score in [0, 1].

    Returns:
        Aggregated verdict dictionary with weighted confidence and component breakdown.
    """
    image_result = image_result or {"score": 0.0, "flags": [], "method": "unknown"}
    metadata_result = metadata_result or {"score": 0.0, "flags": [], "method": "exif"}

    neural_score = image_result.get("raw_scores", {}).get("face_score", 0.5) / 100
    gan_score = image_result.get("raw_scores", {}).get("gan_score", 0.0)
    metadata_score = metadata_result.get("score", 0.0)

    weighted = (
        (neural_score * IMAGE_WEIGHTS["neural"])
        + (gan_score * IMAGE_WEIGHTS["gan"])
        + (metadata_score * IMAGE_WEIGHTS["metadata"])
    )

    all_flags = _combine_flags(image_result, metadata_result)
    result = "FAKE" if weighted >= FAKE_THRESHOLD else "REAL"

    return {
        "result": result,
        "confidence": round(weighted * 100, 1),
        "weighted_score": round(weighted, 4),
        "component_scores": {
            "neural_score": round(neural_score, 4),
            "gan_score": round(gan_score, 4),
            "metadata_score": round(metadata_score, 4),
        },
        "flags": all_flags,
        "file_type": "image",
    }


def aggregate_video_result(
    video_result: Dict[str, Any],
    audio_result: Dict[str, Any],
    metadata_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Aggregate video + audio + metadata detector signals into one final video verdict.

    Args:
        video_result: Video detector output with confidence and frame_timeline.
        audio_result: Audio detector output with score and audio_available.
        metadata_result: Metadata detector output with score in [0, 1].

    Returns:
        Aggregated verdict dictionary with weighted confidence and component breakdown.
    """
    video_result = video_result or {"score": 0.0, "flags": [], "method": "unknown"}
    audio_result = audio_result or {"score": 0.0, "flags": [], "method": "unknown"}
    metadata_result = metadata_result or {"score": 0.0, "flags": [], "method": "exif"}

    neural_score = video_result.get("confidence", 0.0) / 100

    raw_gan_scores = video_result.get("raw_gan_scores", [0.0])
    if raw_gan_scores:
        gan_score = sum(raw_gan_scores) / len(raw_gan_scores)
    else:
        gan_score = 0.0

    audio_score = audio_result.get("score", 0.5) if audio_result.get("audio_available") else 0.5
    metadata_score = metadata_result.get("score", 0.0)

    weighted = (
        (neural_score * VIDEO_WEIGHTS["neural"])
        + (gan_score * VIDEO_WEIGHTS["gan"])
        + (audio_score * VIDEO_WEIGHTS["audio"])
        + (metadata_score * VIDEO_WEIGHTS["metadata"])
    )

    frame_timeline = video_result.get("frame_timeline", [])
    all_flags = _combine_flags(video_result, audio_result, metadata_result)
    result = "FAKE" if weighted >= FAKE_THRESHOLD else "REAL"

    return {
        "result": result,
        "confidence": round(weighted * 100, 1),
        "weighted_score": round(weighted, 4),
        "component_scores": {
            "neural_score": round(neural_score, 4),
            "gan_score": round(gan_score, 4),
            "audio_score": round(audio_score, 4),
            "metadata_score": round(metadata_score, 4),
        },
        "flags": all_flags,
        "file_type": "video",
        "frame_timeline": frame_timeline,
        "total_frames_analyzed": video_result.get("total_frames_analyzed", len(frame_timeline)),
        "fake_frames": video_result.get("fake_frames", 0),
        "fake_ratio": video_result.get("fake_ratio", 0.0),
        "duration_sec": video_result.get("duration_sec"),
    }


def build_threat_summary(aggregated: Dict[str, Any]) -> str:
    """
    Build a short plain-English intelligence summary from aggregated detector output.

    Args:
        aggregated: Aggregated detector result dictionary.

    Returns:
        A 2-3 sentence summary string.
    """
    flag_descriptions = {
        "no_face_detected": "no clear face landmarks were consistently detected",
        "gan_artifacts_detected": "frequency-domain patterns suggest GAN-style synthesis artifacts",
        "high_face_fake_probability": "facial feature analysis indicates likely synthetic generation",
        "analysis_exception": "some detector pipelines encountered processing anomalies",
        "video_open_failed": "the video stream could not be reliably decoded",
        "no_frames_analyzed": "insufficient valid frames were available for robust analysis",
        "audio_unavailable": "audio evidence was unavailable for cross-checking",
        "voice_clone_indicators": "voice characteristics suggest possible cloning or synthesis",
        "metadata_tampering_detected": "file metadata patterns indicate potential tampering",
    }

    result = aggregated.get("result", "REAL")
    confidence = aggregated.get("confidence", 0)
    flags = aggregated.get("flags", [])

    if result == "FAKE":
        top_flags = flags[:2]
        readable_flags = [
            flag_descriptions.get(flag, flag.replace("_", " ")) for flag in top_flags
        ]
        if readable_flags:
            flag_text = "Key indicators include " + "; and ".join(readable_flags) + "."
        else:
            flag_text = "Multiple detector signals indicate elevated manipulation risk."
        return (
            "Analysis indicates this content is likely synthetic or manipulated. "
            f"{flag_text} Confidence: {confidence}%."
        )

    component_scores = aggregated.get("component_scores", {})
    vector_count = len(component_scores) if component_scores else 0
    if vector_count <= 0:
        vector_count = 1

    return (
        "No significant manipulation indicators detected. "
        f"Content appears authentic across {vector_count} detection vectors. "
        f"Confidence: {confidence}%."
    )
