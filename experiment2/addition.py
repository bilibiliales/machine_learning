from pathlib import Path
from PIL import Image
import re

# ============================================================
# 配置区：按需修改这两个路径，不需要 UI
# ============================================================

# 把自己的 JPG/PNG 照片放在这个文件夹中
SOURCE_DIR = Path("dataset/my_faces")

# 转换后的照片保存为新的身份类别 s41
OUTPUT_DIR = Path("dataset/att_faces/s41")

# ORL/AT&T 人脸图像规格：宽 92，高 112
IMAGE_SIZE = (92, 112)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}


def natural_key(path: Path):
    """按文件名中的数字自然排序，例如 photo2 排在 photo10 前面。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"找不到照片目录：{SOURCE_DIR.resolve()}\n"
            f"请创建该目录，并把自己的多张 JPG/PNG 照片放进去。"
        )

    photos = sorted(
        [
            p for p in SOURCE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=natural_key,
    )

    if len(photos) < 2:
        raise ValueError(
            f"至少需要 2 张不同的照片，当前找到 {len(photos)} 张。"
            "请勿把同一张照片同时放入训练集和测试集。"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 只覆盖本脚本生成的 1.pgm、2.pgm...，不删除其他文件。
    for old_file in OUTPUT_DIR.glob("*.pgm"):
        if old_file.stem.isdigit():
            old_file.unlink()

    for index, photo_path in enumerate(photos, start=1):
        output_path = OUTPUT_DIR / f"{index}.pgm"

        with Image.open(photo_path) as image:
            # 转成灰度，并调整为 ORL 数据集的 92×112（宽×高）
            image = image.convert("L")
            image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            image.save(output_path, format="PPM")

        print(f"{photo_path.name} -> {output_path}")

    print()
    print(f"转换完成：共 {len(photos)} 张照片")
    print(f"输出目录：{OUTPUT_DIR.resolve()}")
    print("这些照片现在位于新类别 s41 中。")
    print("请按下方说明修改原实验的数据读取循环，让照片进入训练集和测试集。")


main()
