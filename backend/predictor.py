import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    from backend.anomaly_detector import (
        analyze_image_anomalies,
        analyze_video_anomalies,
        sample_video_frames,
    )
    from backend.config import MODELS_PATH, VIDEO_FORMATS
except ImportError:
    from anomaly_detector import analyze_image_anomalies, analyze_video_anomalies, sample_video_frames  # type: ignore
    from config import MODELS_PATH, VIDEO_FORMATS  # type: ignore


MODEL_SPECS = {
    "resnet50": ("resnet50", 224, "resnet50_model.h5"),
    "efficientnet": ("efficientnet_b3", 300, "efficientnet_model.h5"),
    "vit": ("vit_base_patch16_384", 384, "vit_model.h5"),
}
_MODEL_CACHE: Dict[str, nn.Module] = {}


def _build_model(key: str) -> nn.Module:
    arch, _, _ = MODEL_SPECS[key]
    base = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="avg")
    in_features = base.num_features
    if key == "resnet50":
        head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
    elif key == "efficientnet":
        head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    else:
        head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
        )
    return nn.Sequential(base, head)


def _preprocess_pil(image: Image.Image, input_size: int) -> torch.Tensor:
    tfm = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor()])
    return tfm(image.convert("RGB")).unsqueeze(0)


def _load_ensemble() -> Dict[str, nn.Module]:
    global _MODEL_CACHE
    if _MODEL_CACHE:
        return _MODEL_CACHE
    models = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key, (_, _, filename) in MODEL_SPECS.items():
        model_path = Path(MODELS_PATH) / filename
        if not model_path.exists():
            print(f"[predictor] Skipping missing model: {model_path}")
            continue
        model = _build_model(key)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        models[key] = model
    if not models:
        raise FileNotFoundError(
            f"No model weights found in {MODELS_PATH}. "
            "Expected at least one of resnet50_model.h5 / efficientnet_model.h5 / vit_model.h5"
        )
    _MODEL_CACHE = models
    return _MODEL_CACHE


def _predict_tensor(models: Dict[str, nn.Module], image: Image.Image) -> Tuple[Dict[str, Dict[str, float]], str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torchvision.datasets.ImageFolder uses alphabetical ordering by class name.
    # With classes ["fake", "real"], index mapping is 0->FAKE, 1->REAL.
    labels = {0: "FAKE", 1: "REAL"}
    vote_scores = {"REAL": 0, "FAKE": 0}
    model_votes: Dict[str, Dict[str, float]] = {}

    for key, model in models.items():
        _, input_size, _ = MODEL_SPECS[key]
        tensor = _preprocess_pil(image, input_size).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = labels[pred_idx]
        confidence = float(probs[pred_idx] * 100.0)
        vote_scores[pred_label] += 1
        model_votes[key] = {"label": pred_label, "confidence": confidence}

    final_label = "FAKE" if vote_scores["FAKE"] >= vote_scores["REAL"] else "REAL"
    confidences = [
        details["confidence"]
        for details in model_votes.values()
        if details["label"] == final_label
    ]
    final_conf = float(np.mean(confidences) if confidences else 50.0)
    return model_votes, final_label, final_conf


def _confidence_level(votes_for_winner: int, confidence: float, total_models: int) -> str:
    if total_models >= 3 and votes_for_winner == 3 and confidence >= 90:
        return "HIGH"
    if votes_for_winner >= max(1, (total_models + 1) // 2) and confidence >= 70:
        return "MEDIUM"
    return "LOW"


def predict_file(file_path: str) -> Dict[str, object]:
    start = time.time()
    models = _load_ensemble()
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in VIDEO_FORMATS:
        frames = sample_video_frames(file_path, count=10)
        if not frames:
            raise ValueError(f"Could not decode video: {file_path}")
        middle = frames[len(frames) // 2]
        image = Image.fromarray(cv2.cvtColor(middle, cv2.COLOR_BGR2RGB))
        anomaly = analyze_video_anomalies(frames)
    else:
        image = Image.open(file_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        anomaly = analyze_image_anomalies(bgr)

    model_votes, prediction, confidence = _predict_tensor(models, image)
    votes_for_winner = sum(1 for v in model_votes.values() if v["label"] == prediction)
    confidence_level = _confidence_level(votes_for_winner, confidence, total_models=len(model_votes))

    result = {
        "file_path": file_path,
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "confidence_level": confidence_level,
        "model_votes": {
            name: f"{vote['label']} ({vote['confidence']:.2f}%)"
            for name, vote in model_votes.items()
        },
        "anomaly_scores": {k: round(v, 4) for k, v in anomaly.component_scores.items()},
        "reasoning": anomaly.flags or ["No strong anomaly indicators detected."],
        "processing_time_ms": int((time.time() - start) * 1000),
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict real/fake for one file.")
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()
    output = predict_file(args.file_path)
    print(json.dumps(output, indent=2))
