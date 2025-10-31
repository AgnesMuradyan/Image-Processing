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


def collect_images(input_dir: Path) -> list[Path]:
    exts = ("*.png", "*.jpg")
    paths = []
    for e in exts:
        paths.extend(glob.glob(str(input_dir / e)))
    paths = [Path(p) for p in sorted(paths)]
    return paths


def draw_hist_image(gray: np.ndarray, width: int = 256, height: int = 200) -> np.ndarray:

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.max() + 1e-9)  # normalize [0,1]
    hist_img = np.full((height, width), 255, dtype=np.uint8)

    hvals = (hist * (height - 1)).astype(np.int32)
    for x in range(width):
        y = height - 1 - hvals[x]
        cv2.line(hist_img, (x, height - 1), (x, y), 0, 1)

    cv2.rectangle(hist_img, (0, 0), (width - 1, height - 1), 0, 1)
    return hist_img


def equalize(gray_image: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray_image)


def clahe_apply(gray_image: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_size), int(tile_size)))
    return clahe.apply(gray_image)


def process_image(path: Path, output_dir: Path, clahe_param_sets: list[tuple[float, int]]):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Could not read: {path}")
        return []
    stem = path.stem
    gray_image = to_gray(img_bgr)

    outputs = []

    orig_hist = draw_hist_image(gray_image)
    outputs.append(save(output_dir, stem, "hist_original", orig_hist))

    eq = equalize(gray_image)
    outputs.append(save(output_dir, stem, "equalized", eq))
    eq_hist = draw_hist_image(eq)
    outputs.append(save(output_dir, stem, "hist_equalized", eq_hist))

    for (clip, tile) in clahe_param_sets:
        clahe_img = clahe_apply(gray_image, clip, tile)
        tag = f"clahe_clip{str(clip).replace('.', 'p')}_tile{tile}"
        outputs.append(save(output_dir, stem, tag, clahe_img))

        hist = draw_hist_image(clahe_img)
        outputs.append(save(output_dir, stem, f"hist_{tag}", hist))

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Histogram processing: Equalization and CLAHE (+histogram images).")
    parser.add_argument(
        "--clahe",
        metavar=("CLIP", "TILE"),
        type=float,
        nargs=2,
        action="append",
        default=[[2.0, 8.0], [4.0, 8.0]],
        help="Add CLAHE parameter set: CLIP TILE (tileGridSize=TILE×TILE). Can be passed multiple times."
    )
    args = parser.parse_args()

    clahe_param_sets = [(float(c), int(t)) for c, t in args.clahe]

    input_dir = Path("./images")
    output_dir = Path("./results")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}. Supported extensions: png, jpg")
        return

    print(f"Processing {len(images)} image(s) from '{input_dir}' saving to '{output_dir}'")
    all_outputs = []
    for p in images[:5]:
        outs = process_image(p, output_dir, clahe_param_sets)
        all_outputs.extend(outs)


if __name__ == "__main__":
    main()
