"""
Deepfake image detection module.

This module provides a lightweight, CPU-only image deepfake analysis pipeline intended
for hackathon-grade deployments:

- Face localization via MTCNN (facenet-pytorch)
- Per-face classification using an EfficientNet-B4 backbone (timm)
- A simple frequency-domain heuristic for GAN artifact detection (DCT energy ratio)

The model is constructed and loaded once at import time for performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torch import nn
from torchvision import transforms

# -----------------------------
# Global, module-level setup
# -----------------------------

_DEVICE = torch.device("cpu")

# Face detector (runs on CPU; keep_all=True to score multiple faces)
_mtcnn = MTCNN(
    keep_all=True,
    device=str(_DEVICE),
    post_process=False,
    min_face_size=40,
)

# ImageNet normalization (EfficientNet expects ImageNet stats)
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


def build_model() -> nn.Module:
    """
    Build the image deepfake classifier.

    Creates an EfficientNet-B4 model (pretrained on ImageNet) and replaces its
    classifier head with a small MLP producing 2 logits: [real, fake].

    Returns:
        A PyTorch model in eval mode on CPU.
    """

    model = timm.create_model("efficientnet_b4", pretrained=True)

    # Resolve classifier input features across timm variants
    in_features: Optional[int] = None
    if hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
        in_features = int(model.classifier.in_features)
    elif hasattr(model, "get_classifier"):
        cls = model.get_classifier()
        if hasattr(cls, "in_features"):
            in_features = int(cls.in_features)

    if in_features is None:
        raise RuntimeError("Unable to determine classifier in_features for EfficientNet-B4.")

    head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
    )

    # timm EfficientNet models generally expose `.classifier`
    if hasattr(model, "classifier"):
        model.classifier = head  # type: ignore[assignment]
    else:
        # Fallback for unusual model wrappers
        model.reset_classifier(num_classes=2)  # type: ignore[attr-defined]
        if hasattr(model, "classifier"):
            model.classifier = head  # type: ignore[assignment]

    model.to(_DEVICE)
    model.eval()
    return model


# Load model once at import time
_model = build_model()


def _detect_gan_artifacts(img: Image.Image) -> float:
    """
    Heuristic detector for GAN-like frequency artifacts using DCT energy ratio.

    The idea: some GAN-generated images may exhibit unnaturally elevated high-frequency
    energy. We compute the 2D DCT of the grayscale image and compare energy in the
    high-frequency (bottom-right) quadrant to the low-frequency (top-left) quadrant.

    Args:
        img: PIL image (any mode); will be converted to grayscale.

    Returns:
        A float score in [0.0, 1.0]. Higher implies stronger GAN-like patterns.
    """

    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    # cv2.dct expects float32
    dct = cv2.dct(arr)
    h, w = dct.shape[:2]

    h2 = max(1, h // 2)
    w2 = max(1, w // 2)

    low = dct[:h2, :w2]
    high = dct[h2:, w2:]

    low_energy = float(np.sum(np.abs(low)))
    high_energy = float(np.sum(np.abs(high)))

    ratio = high_energy / (low_energy + 1e-8)

    # Normalize to [0, 1] with a simple scaling if above a threshold
    if ratio > 0.15:
        return float(min(1.0, ratio * 3.0))
    return 0.0


def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Analyze an image for deepfake likelihood.

    Pipeline:
    1) Load image via PIL and convert to RGB.
    2) Detect faces via MTCNN; if none detected, analyze the full image as one "face"
       and add the "no_face_detected" flag.
    3) For each detected face box, crop and classify with the global EfficientNet model.
    4) Compute average face fake probability.
    5) Compute a GAN artifact boost score from the full image using `_detect_gan_artifacts`.
    6) Combine scores: combined = 0.75 * avg_face + 0.25 * gan_boost.
    7) Produce flags and return a structured result.

    Args:
        image_path: Path to an image file.

    Returns:
        A dict containing:
        - result: "FAKE", "REAL", or "ERROR"
        - confidence: combined score as a percentage (0-100), rounded to 1 decimal
        - faces_detected: number of faces detected by MTCNN
        - face_scores: list of per-face fake probabilities, rounded to 1 decimal
        - flags: list of string flags describing notable conditions
        - raw_scores: dict with raw face and gan scores (0-1)

        On exception, returns:
        - result="ERROR", confidence=0, flags=["analysis_exception"], error=str(e)
    """

    try:
        img_path = Path(image_path)
        img = Image.open(img_path).convert("RGB")

        boxes, _ = _mtcnn.detect(img)

        flags: List[str] = []
        face_crops: List[Image.Image] = []

        if boxes is None or len(boxes) == 0:
            flags.append("no_face_detected")
            face_crops = [img]
            faces_detected = 0
        else:
            faces_detected = int(len(boxes))
            w, h = img.size
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                left = max(0, int(round(x1)))
                top = max(0, int(round(y1)))
                right = min(w, int(round(x2)))
                bottom = min(h, int(round(y2)))
                if right <= left or bottom <= top:
                    continue
                face_crops.append(img.crop((left, top, right, bottom)))

            # If all boxes were invalid for any reason, fall back to full image
            if not face_crops:
                flags.append("no_face_detected")
                face_crops = [img]
                faces_detected = 0

        face_scores: List[float] = []
        with torch.no_grad():
            for crop in face_crops:
                x = _transform(crop).unsqueeze(0).to(_DEVICE)
                logits = _model(x)
                probs = torch.softmax(logits, dim=1)
                fake_prob = float(probs[0][1].item())
                face_scores.append(fake_prob)

        avg_face_score = float(np.mean(face_scores)) if face_scores else 0.5

        gan_boost = float(_detect_gan_artifacts(img))
        combined_score = (avg_face_score * 0.75) + (gan_boost * 0.25)

        if gan_boost > 0.3:
            flags.append("gan_artifacts_detected")
        if avg_face_score > 0.7:
            flags.append("high_face_fake_probability")

        result = "FAKE" if combined_score > 0.52 else "REAL"

        return {
            "result": result,
            "confidence": round(combined_score * 100.0, 1),
            "faces_detected": faces_detected,
            "face_scores": [round(s, 1) for s in face_scores],
            "flags": flags,
            "raw_scores": {
                "face_score": avg_face_score,
                "gan_score": gan_boost,
            },
        }
    except Exception as e:  # noqa: BLE001 - explicit requirement to catch all exceptions
        return {
            "result": "ERROR",
            "confidence": 0,
            "flags": ["analysis_exception"],
            "error": str(e),
        }
