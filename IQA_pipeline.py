import os
import cv2

from IQA_frequency import f_analysis, f_detect, f_correction, BRISQUE_THRESHOLD
from IQA_exposition import exp_analysis, exp_correction, EXP_LOW, EXP_HIGH
from IQA_saturation import sat_analysis, sat_correction, SAT_LOW, SAT_HIGH, SAT_GREY

INPUT_DIR = "img/random"
OUTPUT_DIR = "img/corrected"
EXP_MAX_PASSES = 3
SAT_MAX_PASSES = 2


def correct_frequency(img, tags):
    """Score with BRISQUE, then apply the single correction f_detect names.

    Two BRISQUE passes at most, and it is the expensive call: one to decide the
    image needs work, one to check the correction did the job. Naming the defect
    with cheap spatial measures replaces the old blind cascade, which tried both
    corrections in turn and undid its own work.

    An unnamed defect fails the stage rather than being guessed at: BRISQUE
    rejected the image, and applying an arbitrary filter would only game the
    score without making the image better.

    Returns:
        (image, succeeded).
    """
    if f_analysis(img) <= BRISQUE_THRESHOLD:
        return img, True

    mode = f_detect(img)
    if mode is None:
        return img, False

    corrected = f_correction(img, mode)
    tags.append(f"{mode}_corrected")
    return corrected, f_analysis(corrected) <= BRISQUE_THRESHOLD


def correct_exposition(img, tags):
    """Pull the exposure back into the target range.

    Returns:
        (image, succeeded).
    """
    for _ in range(EXP_MAX_PASSES):
        mean = exp_analysis(img)
        if EXP_LOW <= mean <= EXP_HIGH:
            return img, True
        img = exp_correction(img, mean)
        tags.append("underexposed_corrected" if mean < EXP_LOW else "overexposed_corrected")
    mean = exp_analysis(img)
    return img, EXP_LOW <= mean <= EXP_HIGH


def _sat_ok(chroma):
    """True when the chroma is in range, or low enough to be a real greyscale shot."""
    return chroma < SAT_GREY or SAT_LOW <= chroma <= SAT_HIGH


def correct_saturation(img, tags):
    """Pull the saturation back into the target range.

    Runs after the exposure pass: a scalar gain in BGR leaves the channel
    ratios untouched, so it barely moves the chroma, whereas scaling the LAB
    chroma leaves L* untouched. The two corrections are near-orthogonal and
    only interact where one of them clips.

    Returns:
        (image, succeeded).
    """
    for _ in range(SAT_MAX_PASSES):
        chroma = sat_analysis(img)
        if _sat_ok(chroma):
            return img, True
        img = sat_correction(img, chroma)
        tags.append("undersaturated_corrected" if chroma < SAT_LOW else "oversaturated_corrected")
    return img, _sat_ok(sat_analysis(img))


def process(img):
    """Run the full pipeline on a BGR image.

    Returns:
        (image, tags, status).
    """
    tags = []
    img, ok = correct_frequency(img, tags)
    if ok:
        img, ok = correct_exposition(img, tags)
    if ok:
        img, ok = correct_saturation(img, tags)
    return img, tags, "ok" if ok else "low_quality"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            img = cv2.imread(os.path.join(root, filename), cv2.IMREAD_COLOR)
            if img is None:
                continue

            img, tags, status = process(img)

            name, ext = os.path.splitext(filename)
            unique_tags = list(dict.fromkeys(tags))
            suffix = status if status == "low_quality" else ("_".join(unique_tags) if unique_tags else "clean")
            out_path = os.path.join(OUTPUT_DIR, f"{name}_{suffix}{ext}")
            cv2.imwrite(out_path, img)
            print(f"{filename}: {status} [{', '.join(tags) if tags else 'aucune correction'}]")


if __name__ == "__main__":
    main()
