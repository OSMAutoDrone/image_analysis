"""Main inference pipeline. It will look in an image folder or single image, 
run the three stages of the IQA pipeline on each, 
and write a CSV summary of the defects found. Corrected images are written to a separate folder.

The three stages are run in order: exposition, frequency, saturation. 
Each stage may add a defect to the list, and may correct the image in place.
"""
import argparse
import csv
import os

import cv2

from analyse_tools.IQA_frequency import f_detect, f_correction
from analyse_tools.IQA_exposition import exp_analysis, exp_correction, EXP_LOW, EXP_HIGH
from analyse_tools.IQA_saturation import sat_analysis, sat_correction, SAT_LOW, SAT_HIGH, SAT_GREY

INPUT_DIR = "./img/abberations/augmented"
OUTPUT_DIR = "./img/corrected"
REPORT_PATH = "./resultats_IQA.csv"
EXP_MAX_PASSES = 3
SAT_MAX_PASSES = 2

# Line order for the csv summary
DEFECTS = ("blur", "noise", "overexposed", "underexposed",
           "oversaturated", "undersaturated")


def correct_exposition(img, defects, skip_corrections=False):
    """Infer the exposure defect from the mean luminance, then apply the matching correction if skip_correction is False."""
    mean = exp_analysis(img)
    # No defect detected
    if EXP_LOW <= mean <= EXP_HIGH:
        return img
    defects.append("underexposed" if mean < EXP_LOW else "overexposed")
    if skip_corrections:
        return img

    for _ in range(EXP_MAX_PASSES):
        img = exp_correction(img, mean)
        mean = exp_analysis(img)
        if EXP_LOW <= mean <= EXP_HIGH:
            break
    return img


def correct_frequency(img, defects, skip_corrections=False):
    """Infer the frequency defect from the image, then apply the matching correction if skip_correction is False."""
    mode = f_detect(img)
    if mode is None:
        return img
    defects.append(mode)
    return img if skip_corrections else f_correction(img, mode)


def _sat_ok(chroma):
    """True when the chroma is in range, or low enough to be a real greyscale shot."""
    return chroma < SAT_GREY or SAT_LOW <= chroma <= SAT_HIGH


def correct_saturation(img, defects, skip_corrections=False):
    """Infer the saturation defect from the chroma, then pull it back in range.

    Runs after the exposure pass, and genuinely depends on it.
    """
    chroma = sat_analysis(img)
    if _sat_ok(chroma):
        return img
    defects.append("undersaturated" if chroma < SAT_LOW else "oversaturated")
    if skip_corrections:
        return img

    for _ in range(SAT_MAX_PASSES):
        img = sat_correction(img, chroma)
        chroma = sat_analysis(img)
        if _sat_ok(chroma):
            break
    return img


def process(img, skip_corrections=False):
    """Run the three stages on a BGR image. Runs in order: exposition, frequency, saturation. Each stage may add a defect to the list.
    corrections are applied in place if skip_corrections is False, and the final image is returned.

    Returns:
        (image, defects) -- defects named in the order the stages run.
    """
    defects = []
    working = correct_exposition(img, defects)
    working = correct_frequency(working, defects, skip_corrections)
    working = correct_saturation(working, defects, skip_corrections)
    return (img if skip_corrections else working), defects


def collect_paths(input_path, single_image):
    """Return the image paths to process: one file, or a whole tree."""
    if single_image:
        return [input_path]
    paths = []
    for root, _, files in os.walk(input_path):
        paths.extend(os.path.join(root, filename) for filename in files)
    return sorted(paths)


def write_report(report_path, flagged, total):
    """Write one row per defect: which images carry it, how many, what share."""
    folder = os.path.dirname(report_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["defaut", "images", "nombre", "pourcentage"])
        for defect in DEFECTS:
            names = flagged[defect]
            share = 100.0 * len(names) / total if total else 0.0
            writer.writerow([defect, ";".join(names), len(names), f"{share:.1f}"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse, et corrige par defaut, un dossier d'images ou une image seule.")
    parser.add_argument("--input_path", default=INPUT_DIR,
                        help="Dossier d'images, ou fichier image si --single_image.")
    parser.add_argument("--output_path", default=OUTPUT_DIR,
                        help="Dossier ou sont ecrites les images corrigees.")
    parser.add_argument("--report_path", default=REPORT_PATH,
                        help="Chemin du CSV de resultats, ecrit dans tous les cas.")
    parser.add_argument("--skip_corrections", action="store_true",
                        help="N'analyser que : aucune correction, aucune image ecrite.")
    parser.add_argument("--single_image", action="store_true",
                        help="Traiter --input_path comme un fichier image unique.")
    return parser.parse_args()


def run_pipeline():
    args = parse_args()
    paths = collect_paths(args.input_path, args.single_image)
    if not args.skip_corrections:
        os.makedirs(args.output_path, exist_ok=True)

    flagged = {defect: [] for defect in DEFECTS}
    total = 0
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        total += 1

        img, defects = process(img, args.skip_corrections)
        name = os.path.splitext(os.path.basename(path))[0]
        for defect in defects:
            flagged[defect].append(name)

        # Only corrected images are written.
        if defects and not args.skip_corrections:
            cv2.imwrite(os.path.join(args.output_path, os.path.basename(path)), img)
        print(f"{name}: {', '.join(defects) if defects else 'aucun defaut'}")

    write_report(args.report_path, flagged, total)
    print(f"\n{total} images analysees -> {args.report_path}")


if __name__ == "__main__":
    run_pipeline()
