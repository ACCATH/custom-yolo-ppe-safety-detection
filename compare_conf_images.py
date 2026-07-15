from pathlib import Path
import cv2


# 三个预测结果文件夹所在的总目录
BASE_DIR = Path("runs/detect/outputs/predict_conf_compare")

CONF_DIRS = {
    "conf=0.25": BASE_DIR / "conf_0_25",
    "conf=0.35": BASE_DIR / "conf_0_35",
    "conf=0.50": BASE_DIR / "conf_0_5",
}

# 拼接后的图片保存位置
OUTPUT_DIR = Path("outputs/conf_compare_grid")

# 支持的图片后缀
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# 每张图统一缩放到这个高度，方便横向拼接
TARGET_HEIGHT = 500

# 顶部文字栏高度
LABEL_BAR_HEIGHT = 45


def get_image_files(folder: Path):
    """获取某个文件夹下的所有图片文件名"""
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    image_files = []
    for file in folder.iterdir():
        if file.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file.name)

    return set(image_files)


def resize_to_height(image, target_height: int):
    """按高度等比例缩放图片"""
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, target_height))
    return resized


def add_label(image, label: str):
    """给图片顶部添加标签栏"""
    h, w = image.shape[:2]

    label_bar = 255 * \
        image[:LABEL_BAR_HEIGHT, :w].copy()  # 先创建白色背景
    label_bar[:] = (255, 255, 255)

    cv2.putText(
        label_bar,
        label,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )

    result = cv2.vconcat([label_bar, image])
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取三个文件夹中的图片文件名
    all_file_sets = []
    for label, folder in CONF_DIRS.items():
        files = get_image_files(folder)
        all_file_sets.append(files)
        print(f"{label}: {len(files)} images found")

    # 只处理三个文件夹里都存在的同名图片
    common_files = set.intersection(*all_file_sets)
    common_files = sorted(common_files)

    print(f"Common images: {len(common_files)}")

    if not common_files:
        print("No common image files found.")
        return

    for filename in common_files:
        panels = []

        for label, folder in CONF_DIRS.items():
            image_path = folder / filename
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Failed to read image: {image_path}")
                break

            image = resize_to_height(image, TARGET_HEIGHT)
            image = add_label(image, label)
            panels.append(image)

        if len(panels) != len(CONF_DIRS):
            continue

        # 横向拼接：conf=0.25 | conf=0.35 | conf=0.50
        combined = cv2.hconcat(panels)

        output_path = OUTPUT_DIR / f"compare_{filename}"
        cv2.imwrite(str(output_path), combined)

        print(f"Saved: {output_path}")

    print("Done!")


if __name__ == "__main__":
    main()