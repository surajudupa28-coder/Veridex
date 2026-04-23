"""
Audio anomaly detection for deepfake analysis (video inputs).

This module extracts mono 16kHz WAV audio from a video file using ffmpeg, then computes
simple heuristics that can correlate with synthesized / manipulated audio:

- Flat energy profile (low RMS coefficient of variation)
- High spectral flatness (noise-like spectrum)
- Low ZCR variance (unnaturally smooth temporal behavior)

These are lightweight signals intended for hackathon-grade scoring, not a definitive
forensic audio classifier.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import librosa
import numpy as np


def extract_audio(video_path: str) -> Optional[str]:
    """
    Extract audio from a video file to a temporary WAV file.

    Uses ffmpeg to extract:
    - mono channel (-ac 1)
    - 16kHz sample rate (-ar 16000)
    - no video (-vn)

    Command:
        ffmpeg -i video_path -ac 1 -ar 16000 -vn temp_path -y -loglevel quiet

    Args:
        video_path: Path to the input video.

    Returns:
        The path to the temporary WAV file on success, or None if extraction fails
        (e.g., ffmpeg error or no audio track).
    """

    video = str(Path(video_path))

    tmp_fd: Optional[int] = None
    tmp_path: Optional[str] = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)

        cmd: List[str] = [
            "ffmpeg",
            "-i",
            video,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            tmp_path,
            "-y",
            "-loglevel",
            "quiet",
        ]

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return None

        # If ffmpeg "succeeded" but produced an empty file, treat as no audio.
        try:
            if os.path.getsize(tmp_path) <= 44:  # ~WAV header size
                os.remove(tmp_path)
                return None
        except OSError:
            return None

        return tmp_path
    except Exception:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return None


def analyze_audio(video_path: str) -> Dict[str, Any]:
    """
    Analyze audio anomalies in a video file for deepfake detection.

    Logic:
    1) Extract audio to a temporary WAV using ffmpeg.
    2) Load with librosa at 16kHz.
    3) Compute heuristic features and flags:
       - Energy coefficient of variation (flat energy profile)
       - Mean spectral flatness (noise-like spectrum)
       - ZCR variance (temporal smoothness)
    4) Score: start at 0.0, add 0.25 for each anomaly flag, cap at 1.0.

    Args:
        video_path: Path to the input video file.

    Returns:
        On success:
            {
              "audio_available": True,
              "score": <0..1>,
              "flags": [...],
              "features": {
                "energy_cv": ...,
                "spectral_flatness": ...
              }
            }
        If no audio track:
            {"audio_available": False, "score": 0.5, "flags": ["no_audio_track"]}
        On exception:
            {"audio_available": False, "score": 0.5, "flags": ["audio_analysis_error"], "error": "..."}
    """

    try:
        wav_path = extract_audio(video_path)
        if wav_path is None:
            return {"audio_available": False, "score": 0.5, "flags": ["no_audio_track"]}

        try:
            y, sr = librosa.load(wav_path, sr=16000)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        if y is None or sr is None or len(y) < sr:
            return {
                "audio_available": True,
                "score": 0.5,
                "flags": ["audio_too_short"],
            }

        flags: List[str] = []

        # FEATURE 1 — Energy variance (RMS coefficient of variation)
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        rms_mean = float(np.mean(rms)) if rms.size else 0.0
        rms_std = float(np.std(rms)) if rms.size else 0.0
        cv = float(rms_std / (rms_mean + 1e-8))
        if cv < 0.3:
            flags.append("flat_energy_profile")

        # FEATURE 2 — Spectral flatness (mean)
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        mean_flatness = float(np.mean(flatness)) if flatness.size else 0.0
        if mean_flatness > 0.25:
            flags.append("high_spectral_flatness")

        # FEATURE 3 — Zero crossing rate anomaly (low variance)
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr_std = float(np.std(zcr)) if zcr.size else 0.0
        if zcr_std < 0.05:
            flags.append("low_zcr_variance")

        score = min(1.0, 0.25 * float(len(flags)))

        return {
            "audio_available": True,
            "score": round(score, 2),
            "flags": flags,
            "features": {
                "energy_cv": round(cv, 3),
                "spectral_flatness": round(mean_flatness, 3),
            },
        }
    except Exception as e:  # noqa: BLE001 - explicit requirement to catch all exceptions
        return {
            "audio_available": False,
            "score": 0.5,
            "flags": ["audio_analysis_error"],
            "error": str(e),
        }
