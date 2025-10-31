import argparse
from pathlib import Path
import glob
import cv2
import numpy as np


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if len(img_bgr.shape) == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def save(output_dir: Path, stem: str, suffix: str, img: np.ndarray):
    file_name = output_dir / f"{stem}__{suffix}.png"
    cv2.imwrite(str(file_name), img)
    return file_name


def binary_threshold(gray_image: np.ndarray, threshold: int = 127) -> np.ndarray:
    x, bw = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return bw


def contrast_stretch(gray_image: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    low = np.percentile(gray_image, low_pct)
    high = np.percentile(gray_image, high_pct)
    if high <= low:
        low = float(np.min(gray_image))
        high = float(np.max(gray_image))
        if high == low:
            return np.zeros_like(gray_image)
    stretched = (gray_image.astype(np.float32) - low) * (255.0 / (high - low))
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    return stretched


def negative(gray_image: np.ndarray) -> np.ndarray:
    return 255 - gray_image


def log_transform(gray_image: np.ndarray) -> np.ndarray:
    gray_f = gray_image.astype(np.float32) / 255.0 # [0,1] normalizacum
    eps = 1e-6  # log(0)-ic xusapelu hamar
    log_img = np.log1p(gray_f + eps)
    log_img /= np.log1p(1.0 + eps)  # normalizacum
    log_img = np.clip(log_img * 255.0, 0, 255).astype(np.uint8)  # [0,1] -> [0,255]
    return log_img


def gamma_transform(gray_image: np.ndarray, gamma: float) -> np.ndarray:
    gray_f = gray_image.astype(np.float32) / 255.0
    out = np.power(gray_f, gamma)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def process_image(path: Path, output_dir: Path, threshold: int, gammas: list[float]):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Could not read: {path}")
        return []
    stem = path.stem
    gray_image = to_gray(img_bgr)

    outputs = []
    bw = binary_threshold(gray_image, threshold=threshold)
    outputs.append(save(output_dir, stem, f"binary_threshold{threshold}", bw))

    cs = contrast_stretch(gray_image, 2.0, 98.0)
    outputs.append(save(output_dir, stem, "contrast_stretch_2_98", cs))

    neg = negative(gray_image)
    outputs.append(save(output_dir, stem, "negative", neg))

    log_img = log_transform(gray_image)
    outputs.append(save(output_dir, stem, "log", log_img))

    for g in gammas:
        gamma_img = gamma_transform(gray_image, g)
        tag = f"gamma_{g}".replace('.', 'p')
        outputs.append(save(output_dir, stem, tag, gamma_img))

    return outputs


def collect_images(input_dir: Path) -> list[Path]:
    exts = ("*.png", "*.jpg")
    paths = []
    for e in exts:
        paths.extend(glob.glob(str(input_dir / e)))
    paths = [Path(p) for p in sorted(paths)]
    return paths


def main():
    parser = argparse.ArgumentParser(description="Apply basic intensity transformations using OpenCV (cv2).")
    parser.add_argument("--threshold", type=int, default=127, help="Global binary threshold value in [0,255] (default: 127).")
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.6, 2.2], help="List of gamma values. Include <1 and >1 (default: 0.6 2.2).")
    args = parser.parse_args()

    input_dir = Path('./images')
    output_dir = Path('./results')

    images = collect_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}. Supported extensions: png, jpg, jpeg, bmp, tif, tiff")
        return

    print(f"Processing {len(images)} image(s) from '{input_dir}' saving to '{output_dir}'")
    all_outputs = []
    for p in images:
        outs = process_image(p, output_dir, args.threshold, args.gamma)
        all_outputs.extend(outs)


if __name__ == "__main__":
    main()
