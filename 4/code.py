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
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = output_dir / f"{stem}__{suffix}.png"
    cv2.imwrite(str(file_name), img)
    return file_name


def collect_images(input_dir: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(str(input_dir / e)))
    paths = [Path(p) for p in sorted(paths)]
    return paths


def box_filter(gray_image: np.ndarray, ksize: int) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.blur(gray_image, (k, k))


def gaussian_filter(gray_image: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(gray_image, (ksize, ksize), sigma)


def median_filter(gray_image: np.ndarray, ksize: int) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    if k < 3:
        k = 3
    return cv2.medianBlur(gray_image, k)


def laplacian_sharpen(gray_image: np.ndarray, ksize: int) -> np.ndarray:
    lap = cv2.Laplacian(gray_image, ddepth=cv2.CV_16S, ksize=ksize, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    lap = cv2.convertScaleAbs(lap)
    sharpened = cv2.addWeighted(gray_image, 1.0, lap, 1.0, 0)
    return sharpened


def unsharp_mask(gray_image: np.ndarray, gksize: int, sigma: float, amount: float) -> np.ndarray:
    k = gksize if gksize % 2 == 1 else gksize + 1
    blurred = cv2.GaussianBlur(gray_image, (k, k), sigmaX=sigma, sigmaY=sigma)
    mask = cv2.subtract(gray_image, blurred)
    sharpened = cv2.addWeighted(gray_image, 1.0, mask, amount, 0)
    return sharpened


def sobel_magnitude(gray_image: np.ndarray, ksize: int) -> np.ndarray:
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    if mag.max() > 0:
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def process_image(path: Path, results_root: Path, args):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Could not read: {path}")
        return []

    stem = path.stem
    gray = to_gray(img_bgr)
    outputs = []

    base_dir = results_root / stem

    ks_359 = [3, 5, 9]
    lap_k = [1, 3, 5]
    unsharp_sets = [(3, 1.0, 0.8), (5, 1.0, 1.0), (9, 1.5, 1.2)]

    for k in ks_359:
        out = box_filter(gray, k)
        outputs.append(save(base_dir / "box", stem, f"box_{k}x{k}", out))

    for k in ks_359:
        out = gaussian_filter(gray, k, args.gaussian_sigma)
        outputs.append(save(base_dir / "gaussian", stem, f"gaussian_{k}x{k}", out))

    for k in ks_359:
        out = median_filter(gray, k)
        outputs.append(save(base_dir / "median", stem, f"median_{k}x{k}", out))

    for k in lap_k:
        out = laplacian_sharpen(gray, k)
        outputs.append(save(base_dir / "laplacian", stem, f"laplacian_k{k}", out))

    for (k, s, a) in unsharp_sets:
        out = unsharp_mask(gray, k, s, a)
        tag = f"unsharp_k{k}_s{str(s).replace('.','p')}_a{str(a).replace('.','p')}"
        outputs.append(save(base_dir / "unsharp", stem, tag, out))

    for k in [1, 3, 5]:
        mag = sobel_magnitude(gray, k)
        outputs.append(save(base_dir / "sobel", stem, f"sobel_mag_k{k}", mag))

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Spatial filters with 3 parameter variants each; save as image/method/3 images.")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    args = parser.parse_args()

    input_dir = Path("./images")
    results_root = Path("./results")

    images = collect_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}. Supported extensions: png, jpg, jpeg, bmp, tif, tiff")
        return

    print(f"Processing {len(images[:5])} image(s) from '{input_dir}' saving to '{results_root}'")
    all_outputs = []
    for p in images[:5]:
        outs = process_image(p, results_root, args)
        all_outputs.extend(outs)


if __name__ == "__main__":
    main()
