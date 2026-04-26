from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None


@dataclass
class AnomalyResult:
    score: float
    flags: List[str]
    component_scores: Dict[str, float]


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _frequency_anomaly(gray: np.ndarray) -> float:
    fft = np.fft.fft2(gray)
    magnitude = np.log1p(np.abs(np.fft.fftshift(fft)))
    h, w = magnitude.shape
    center = magnitude[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    outer_mean = float(np.mean(magnitude))
    center_mean = float(np.mean(center))
    ratio = (outer_mean - center_mean) / (outer_mean + 1e-6)
    return _clamp01(0.5 + ratio)


def _facial_landmark_anomaly(frame_rgb: np.ndarray) -> float:
    if mp is None:
        return 0.5
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    result = face_mesh.process(frame_rgb)
    face_mesh.close()
    if not result.multi_face_landmarks:
        return 0.7
    landmarks = result.multi_face_landmarks[0].landmark
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    eye_diff = abs(left_eye.y - right_eye.y)
    mouth_diff = abs(mouth_left.y - mouth_right.y)
    return _clamp01((eye_diff + mouth_diff) * 6.0)


def _color_space_anomaly(frame_bgr: np.ndarray) -> float:
    ycbcr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycbcr)
    y_std = float(np.std(y))
    chroma_std = float((np.std(cr) + np.std(cb)) / 2.0)
    score = abs(chroma_std - y_std) / (y_std + 1e-6)
    return _clamp01(score / 2.0)


def _lighting_anomaly(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    uniformity = float(np.std(magnitude) / (np.mean(magnitude) + 1e-6))
    return _clamp01(uniformity / 3.0)


def _optical_flow_anomaly(frames_bgr: List[np.ndarray]) -> float:
    if len(frames_bgr) < 2:
        return 0.5
    mags = []
    prev_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    for frame in frames_bgr[1:]:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(float(np.mean(mag)))
        prev_gray = curr_gray
    if not mags:
        return 0.5
    return _clamp01(float(np.std(mags) / (np.mean(mags) + 1e-6)))


def _flags(scores: Dict[str, float]) -> List[str]:
    labels = {
        "frequency_anomaly": "Frequency domain shows unnatural artifacts",
        "facial_landmark_anomaly": "Facial landmarks have asymmetric patterns",
        "color_space_anomaly": "Color space discontinuities detected",
        "optical_flow_anomaly": "Optical flow shows jerky motion",
        "lighting_anomaly": "Lighting and shadow consistency is weak",
    }
    return [labels[key] for key, val in scores.items() if val >= 0.65]


def analyze_image_anomalies(frame_bgr: np.ndarray) -> AnomalyResult:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    scores = {
        "frequency_anomaly": _frequency_anomaly(gray),
        "facial_landmark_anomaly": _facial_landmark_anomaly(rgb),
        "color_space_anomaly": _color_space_anomaly(frame_bgr),
        "optical_flow_anomaly": 0.0,
        "lighting_anomaly": _lighting_anomaly(gray),
    }
    score = float(np.mean(list(scores.values())))
    return AnomalyResult(score=score, flags=_flags(scores), component_scores=scores)


def analyze_video_anomalies(frames_bgr: List[np.ndarray]) -> AnomalyResult:
    if not frames_bgr:
        empty = {
            "frequency_anomaly": 0.0,
            "facial_landmark_anomaly": 0.0,
            "color_space_anomaly": 0.0,
            "optical_flow_anomaly": 0.0,
            "lighting_anomaly": 0.0,
        }
        return AnomalyResult(score=0.0, flags=[], component_scores=empty)

    base = analyze_image_anomalies(frames_bgr[len(frames_bgr) // 2]).component_scores
    base["optical_flow_anomaly"] = _optical_flow_anomaly(frames_bgr)
    score = float(np.mean(list(base.values())))
    return AnomalyResult(score=score, flags=_flags(base), component_scores=base)


def sample_video_frames(video_path: str, count: int = 10) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, max(0, total - 1), count, dtype=int).tolist()
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    cap.release()
    return frames
