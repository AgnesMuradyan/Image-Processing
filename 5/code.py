import argparse
from pathlib import Path
import glob
import csv
import cv2
import numpy as np


def save(output_dir, stem, suffix, img):
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"{stem}__{suffix}.png"
    cv2.imwrite(str(fname), img)
    return fname


def list_images_recursive(root):
    exts = ("**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.bmp", "**/*.tif", "**/*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(str(Path(root) / e), recursive=True))
    return [Path(p) for p in sorted(paths)]


def filt_box(gray, k): return cv2.blur(gray, (k, k))
def filt_gaussian(gray, k, sigma): return cv2.GaussianBlur(gray, (k, k), sigma)
def filt_median(gray, k): return cv2.medianBlur(gray, k)
def filt_bilateral(gray, d, sc, ss): return cv2.bilateralFilter(gray, d, sc, ss)
def filt_nlm(gray, h, t, s): return cv2.fastNlMeansDenoising(gray, None, h, t, s)


def estimate_noise_variance(img):
    img_f = img.astype(np.float32)
    k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    hp = cv2.filter2D(img_f, -1, k)
    sigma2 = np.median(np.abs(hp))**2
    return sigma2 if sigma2 > 1e-6 else 1e-6


def filt_wiener_local(gray, k):
    y = gray.astype(np.float32)
    mu = cv2.boxFilter(y, -1, (k, k))
    mu2 = mu * mu
    m2 = cv2.boxFilter(y * y, -1, (k, k))
    var = np.maximum(m2 - mu2, 0)
    nv = estimate_noise_variance(gray)
    g = np.clip((var - nv) / (var + 1e-6), 0, 1)
    out = mu + g * (y - mu)
    return np.clip(out, 0, 255).astype(np.uint8)


def psnr(a, b): return float(cv2.PSNR(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", default="./images")
    parser.add_argument("--gt-dir", default="./ground_truth")
    parser.add_argument("--results", default="./results")
    parser.add_argument("--gaussian-sigma", type=float, default=1.0)
    parser.add_argument("--nlm-h", type=float, default=10.0)
    parser.add_argument("--nlm-template", type=int, default=7)
    parser.add_argument("--nlm-search", type=int, default=21)
    parser.add_argument("--bilateral-d", type=int, default=9)
    parser.add_argument("--bilateral-sigma-color", type=float, default=75.0)
    parser.add_argument("--bilateral-sigma-space", type=float, default=75.0)
    args = parser.parse_args()

    gt_dir, input_dir, out_root = Path(args.gt_dir), Path(args.images_root), Path(args.results)
    gt_paths = list_images_recursive(gt_dir)
    gt_by_stem = {p.stem: p for p in gt_paths}
    noisy_paths = list_images_recursive(input_dir)
    if not noisy_paths:
        print("No images found.")
        return

    kernels = [3, 5, 9]
    rows = []

    for noisy_path in noisy_paths:
        stem = noisy_path.stem
        gt_path = gt_by_stem.get(stem)
        if gt_path is None:
            print(f"No GT for {stem}")
            continue

        noisy = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if noisy is None or gt is None or noisy.shape != gt.shape:
            print(f"Skipping {stem}")
            continue

        base_dir = out_root / stem
        save(base_dir / "input", stem, "noisy", noisy)
        save(base_dir / "input", stem, "gt", gt)

        methods = {
            "box": lambda g, k: filt_box(g, k),
            "gaussian": lambda g, k: filt_gaussian(g, k, args.gaussian_sigma),
            "median": lambda g, k: filt_median(g, k),
            "bilateral": lambda g, k: filt_bilateral(g, args.bilateral_d, args.bilateral_sigma_color, args.bilateral_sigma_space),
            "wiener": lambda g, k: filt_wiener_local(g, k),
            "nlm": lambda g, k: filt_nlm(g, args.nlm_h, args.nlm_template, args.nlm_search)
        }

        for method_name, func in methods.items():
            for k in kernels:
                res = func(noisy, k)
                save(base_dir / method_name, stem, f"{method_name}_{k}x{k}", res)
                rows.append((stem, f"{method_name}_{k}x{k}", psnr(gt, res)))

    csv_path = out_root / "psnr_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_stem", "method", "psnr"])
        w.writerows(rows)

    means = {}
    for _, method, val in rows:
        means.setdefault(method, []).append(val)
    mean_psnr = {m: np.mean(v) for m, v in means.items()}

    summary = out_root / "PSNR_summary.txt"
    with open(summary, "w") as f:
        for m, v in mean_psnr.items():
            f.write(f"{m:25s}: {v:.3f} dB\n")

    print(f"Done. Results in {out_root}")


if __name__ == "__main__":
    main()
