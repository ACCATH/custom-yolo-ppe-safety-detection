from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "runs/detect/outputs/train/ppe_yolov8n/weights/best.pt"
DATA_YAML = "datasets/construction_site_safety/data.yaml"
PROJECT_DIR = "outputs/evaluate"
RUN_NAME = "ppe_eval"


def main():
    model_path = Path(MODEL_PATH)
    data_yaml_path = Path(DATA_YAML)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    model = YOLO(str(model_path))

    model.val(
        data=str(data_yaml_path),
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True
    )


if __name__ == "__main__":
    main()