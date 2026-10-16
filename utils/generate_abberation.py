"""Build a labelled benchmark set of degraded images for the IQA pipeline. Used mainly to generate test data.
"""

import csv
import os

import cv2
import numpy as np

from analyse_tools.IQA_exposition import exp_analysis
from analyse_tools.IQA_saturation import sat_analysis, sat_scale

INPUT_DIR = "../AI_analysis/data/base/osm/img"
OUTPUT_DIR = "img/abberations/augmented"
MANIFEST = "manifest.csv"

# Re-encoding at high quality keeps JPEG artifacts out of the BRISQUE score,
# where they would otherwise be blamed on the injected defect.
# BRISQUE is not currently used in the pipeline, but it used to be. This is an artifact that might not be currently usefull.
JPEG_QUALITY = 95

SEED = 42

DEFECTS = ("clean", "blur", "noise", "overexposed", "underexposed",
           "oversaturated", "undersaturated")

# Intensity ranges, kept within what a real camera plausibly produces while
# still crossing the detection thresholds on most of the source set.
BLUR_SIGMA = (1.0, 3.5)              # pixels: mild to obvious defocus
NOISE_SIGMA = (8.0, 25.0)            # grey levels: roughly ISO 1600 to ISO 12800
OVEREXPOSE_GAIN = (1.40, 2)         # x1.7 clears EXP_HIGH on 77% of the set
UNDEREXPOSE_GAIN = (0.30, 0.45)      # x0.45 clears EXP_LOW on 97%
OVERSATURATE_GAIN = (3.00, 6.00)     # LAB clipping makes anything below x3 too weak
UNDERSATURATE_GAIN = (0.30, 0.48)    # floor kept above SAT_GREY to avoid the greyscale guard


def apply_blur(img_arr, rng):
    """Defocus the image with a Gaussian kernel sized from the drawn sigma."""
    sigma = rng.uniform(*BLUR_SIGMA)
    ksize = 2 * int(round(3.0 * sigma)) + 1
    return cv2.GaussianBlur(img_arr, (ksize, ksize), sigma), sigma


def apply_noise(img_arr, rng):
    """Add zero-mean Gaussian sensor noise, drawn independently per channel.

    Per-channel noise also raises the measured chroma, so a noisy image can
    legitimately trip the saturation stage as well.
    """
    sigma = rng.uniform(*NOISE_SIGMA)
    noisy = img_arr.astype(np.float32) + rng.normal(0.0, sigma, img_arr.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8), sigma


def apply_exposure(img_arr, rng, over):
    """Scale every channel, mirroring exp_correction's gain and its clipping."""
    gain = rng.uniform(*(OVEREXPOSE_GAIN if over else UNDEREXPOSE_GAIN))
    return cv2.convertScaleAbs(img_arr, alpha=gain), gain


def apply_saturation(img_arr, rng, over):
    """Scale the LAB chroma, mirroring sat_correction."""
    gain = rng.uniform(*(OVERSATURATE_GAIN if over else UNDERSATURATE_GAIN))
    return sat_scale(img_arr, gain), gain


def degrade(img_arr, defect, rng):
    """Apply one defect class.

    Returns:
        (image, intensity), the intensity being 1.0 for an untouched image.
    """
    if defect == "blur":
        return apply_blur(img_arr, rng)
    if defect == "noise":
        return apply_noise(img_arr, rng)
    if defect in ("overexposed", "underexposed"):
        return apply_exposure(img_arr, rng, defect == "overexposed")
    if defect in ("oversaturated", "undersaturated"):
        return apply_saturation(img_arr, rng, defect == "oversaturated")
    return img_arr, 1.0


def collect_paths():
    """Return every file under INPUT_DIR, sorted so a rerun draws the same values."""
    paths = []
    for root, _, files in os.walk(INPUT_DIR):
        paths.extend(os.path.join(root, filename) for filename in files)
    return sorted(paths)


def assign_defects(count, rng):
    """Deal the classes round-robin, then shuffle.

    Drawing each class independently would only balance them in expectation;
    dealing them keeps every class within one image of the others, so each
    per-class hit rate rests on the same sample size.
    """
    defects = [DEFECTS[i % len(DEFECTS)] for i in range(count)]
    rng.shuffle(defects)
    return defects


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    paths = collect_paths()
    defects = assign_defects(len(paths), rng)

    manifest_path = os.path.join(OUTPUT_DIR, MANIFEST)
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "output", "defect", "intensity",
                         "mean_before", "mean_after",
                         "chroma_before", "chroma_after"])

        for path, defect in zip(paths, defects):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            out_img, intensity = degrade(img, defect, rng)

            relative = os.path.relpath(path, INPUT_DIR)
            name, ext = os.path.splitext(relative.replace(os.sep, "_"))
            out_name = f"{name}_{defect}{ext}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), out_img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            writer.writerow([relative, out_name, defect, f"{intensity:.3f}",
                             f"{exp_analysis(img):.1f}", f"{exp_analysis(out_img):.1f}",
                             f"{sat_analysis(img):.2f}", f"{sat_analysis(out_img):.2f}"])
            print(f"{relative}: {defect} (x{intensity:.2f})")

    print(f"\n{len(paths)} images -> {OUTPUT_DIR}, verite terrain dans {manifest_path}")


if __name__ == "__main__":
    main()
