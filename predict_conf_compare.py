from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = "weights/ppe_yolov8n_best_map077.pt"
SOURCE_DIR = "datasets/construction_site_safety/test/images"
OUTPUT_DIR = "outputs/predict_conf_compare"

CONF_LIST = [0.25, 0.35, 0.50]


def main():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    for conf in CONF_LIST:
        run_name = f"conf_{str(conf).replace('.', '_')}"

        model.predict(
            source=SOURCE_DIR,
            save=True,
            conf=conf,
            project=OUTPUT_DIR,
            name=run_name,
            exist_ok=True
        )

        print(f"Prediction finished with conf={conf}")
        print(f"Saved to: {OUTPUT_DIR}/{run_name}")


if __name__ == "__main__":
    main()