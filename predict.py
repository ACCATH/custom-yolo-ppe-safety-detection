from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = "weights/ppe_yolov8n_best_map077.pt"
SOURCE_DIR = "datasets/construction_site_safety/test/images"
OUTPUT_DIR = "outputs/predict"
RUN_NAME = "ppe_predictions"



def main():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    model.predict(
        source=SOURCE_DIR,
        save=True,
        conf=0.25,
        project=OUTPUT_DIR,
        name=RUN_NAME,
        exist_ok=True
    )


if __name__ == "__main__":
    main()