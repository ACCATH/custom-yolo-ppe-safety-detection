from ultralytics import YOLO
from pathlib import Path


DATA_YAML = "datasets/construction_site_safety/data.yaml"
MODEL_NAME = "yolov8s.pt"

EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 4
DEVICE = 0

PROJECT_DIR = "outputs/train"
RUN_NAME = "ppe_yolov8s_e50_img640"


def main():
    data_yaml_path = Path(DATA_YAML)

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(data_yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True
    )

    


if __name__ == "__main__":
    main()