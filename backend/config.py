from pathlib import Path

# Data paths (CRITICAL - DO NOT CHANGE PATHS)
DATASET_REAL_PATH = "backend/dataset/real"
DATASET_FAKE_PATH = "backend/dataset/fake"
VIDEOS_REAL_PATH = "backend/videos/real"
VIDEOS_FAKE_PATH = "backend/videos/fake"
COMBINED_DATASET_PATH = "backend/dataset/dataset_combined"
MODELS_PATH = "backend/models"
REPORT_PATH = "backend/report"

# Image settings
IMAGE_SIZE = (224, 224)  # For ResNet-50
IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".flv"]
FRAMES_PER_VIDEO = 10

# Model settings
MODEL_TYPE = "ensemble"  # Options: resnet50, efficientnet, vit, ensemble
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Training settings
EARLY_STOPPING_PATIENCE = 5
CLASS_WEIGHT_BALANCING = True
DATA_AUGMENTATION = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3

# Prediction settings
CONFIDENCE_THRESHOLD = 0.7
ANOMALY_DETECTION = True

# Output settings
GENERATE_HTML_REPORT = True
SAVE_PREDICTIONS_CSV = True
SAVE_VISUALIZATIONS = True


def ensure_dirs() -> None:
    for path in (
        COMBINED_DATASET_PATH,
        MODELS_PATH,
        REPORT_PATH,
        f"{COMBINED_DATASET_PATH}/frames",
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
