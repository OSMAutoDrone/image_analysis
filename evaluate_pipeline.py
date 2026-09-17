"""Score IQA_pipeline against the labelled set produced by generate_abberation.

Writes a timestamped run folder holding results.csv, confusion.csv, summary.txt
and, for every image, a source | degraded | corrected triptych filed under the
outcome it got, so a failure can be read by eye as well as by number.
"""

import collections
import csv
import os
import time

import cv2
import numpy as np

from IQA_pipeline import process
from IQA_frequency import f_analysis, BRISQUE_THRESHOLD
from IQA_exposition import exp_analysis, EXP_LOW, EXP_HIGH
from IQA_saturation import sat_analysis, SAT_LOW, SAT_HIGH, SAT_GREY
from generate_abberation import INPUT_DIR as SOURCE_DIR, OUTPUT_DIR as AUGMENTED_DIR, MANIFEST, DEFECTS

RUNS_DIR = "runs"
PANEL_WIDTH = 420

# What the pipeline can say, in the order it decides: a frequency failure stops
# the later stages from running at all, so a frequency label always comes first.
PREDICTIONS = ("blur", "noise", "overexposed", "underexposed",
               "oversaturated", "undersaturated")

# Confusion matrix columns. "clean" is the default prediction, the one given to
# an image the pipeline corrected nothing on.
COLUMNS = ("clean",) + PREDICTIONS

BUCKETS = ("ok", "ko_detection", "ko_classification", "ko_correction", "false_positive")


def predict(tags):
    """Read back the defects the pipeline acted on, from its correction tags.

    correct_frequency commits to a single frequency label, so blur and noise
    are mutually exclusive here and the matrix can tell them apart.
    """
    labels = [label for label in PREDICTIONS if f"{label}_corrected" in tags]
    return labels or ["clean"]


def check(img_arr):
    """Measure the three gates and name the ones the image still fails."""
    brisque, mean, chroma = f_analysis(img_arr), exp_analysis(img_arr), sat_analysis(img_arr)
    reasons = []
    if brisque > BRISQUE_THRESHOLD:
        reasons.append(f"brisque {brisque:.0f} > {BRISQUE_THRESHOLD}")
    if not EXP_LOW <= mean <= EXP_HIGH:
        reasons.append(f"mean {mean:.0f} hors [{EXP_LOW},{EXP_HIGH}]")
    if not (chroma < SAT_GREY or SAT_LOW <= chroma <= SAT_HIGH):
        reasons.append(f"chroma {chroma:.1f} hors [{SAT_LOW},{SAT_HIGH}]")
    return brisque, mean, chroma, " ; ".join(reasons)


def detected(truth, labels):
    """True when the pipeline flagged a defect on the same axis as the truth."""
    if truth == "clean":
        return labels == ["clean"]
    if truth in ("blur", "noise"):
        return "blur" in labels or "noise" in labels
    return truth in labels


def outcome(truth, labels, ok):
    """Bucket one image; the bucket names the folder its triptych lands in."""
    if truth == "clean":
        return "ok" if labels == ["clean"] else "false_positive"
    if labels == ["clean"]:
        return "ko_detection"
    if not detected(truth, labels):
        return "ko_classification"
    return "ok" if ok else "ko_correction"


def panel(img_arr, title, subtitle):
    """Scale one image to the panel width and caption it underneath."""
    height, width = img_arr.shape[:2]
    scaled = cv2.resize(img_arr, (PANEL_WIDTH, round(height * PANEL_WIDTH / width)))
    caption = np.full((44, PANEL_WIDTH, 3), 255, np.uint8)
    cv2.putText(caption, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(caption, subtitle, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 70, 70), 1, cv2.LINE_AA)
    return np.vstack([scaled, caption])


def rate_line(label, group):
    """Format one row of the per-category detection/correction table."""
    return (f"{label:16}{len(group):5}"
            f"{sum(r['detected'] for r in group) / len(group):9.0%}"
            f"{sum(r['corrected'] for r in group) / len(group):9.0%}")


def report(records, matrix):
    """Build the summary printed in the terminal and saved beside the results."""
    lines = ["=== Detection et correction par categorie ===",
             f"{'categorie':16}{'n':>5}{'detecte':>10}{'corrige':>10}"]
    for truth in DEFECTS:
        group = [r for r in records if r["truth"] == truth]
        if group:
            lines.append(rate_line(truth, group))
    lines.append(rate_line("TOTAL", records))

    lines += ["", "=== Matrice de confusion (lignes = verite, colonnes = prediction) ==="]
    lines.append(f"{'':16}" + "".join(f"{column[:12]:>14}" for column in COLUMNS))
    for truth in DEFECTS:
        lines.append(f"{truth:16}" + "".join(f"{matrix[(truth, column)]:14}" for column in COLUMNS))

    lines += ["", "=== Repartition des issues ==="]
    buckets = collections.Counter(r["outcome"] for r in records)
    for bucket in BUCKETS:
        lines.append(f"  {bucket:20}{buckets[bucket]:5} ({buckets[bucket] / len(records):.0%})")

    lines += ["", "=== Premier seuil bloquant, sur les images non corrigees ==="]
    blocked = collections.Counter(r["blocked_on"] for r in records if not r["corrected"])
    for gate, count in blocked.most_common():
        lines.append(f"  {gate:20}{count:5}")
    lines.append("  (les etages s'enchainent sous condition : un echec amont empeche")
    lines.append("   les seuils suivants d'etre seulement tentes)")
    return lines


def main():
    manifest_path = os.path.join(AUGMENTED_DIR, MANIFEST)
    with open(manifest_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{manifest_path} est vide : lance d'abord generate_abberation.py")

    run_dir = os.path.join(RUNS_DIR, time.strftime("eval_%Y%m%d_%H%M%S"))
    records = []
    started = time.time()

    for index, row in enumerate(rows, 1):
        truth = row["defect"]
        source = cv2.imread(os.path.join(SOURCE_DIR, row["source"]), cv2.IMREAD_COLOR)
        degraded = cv2.imread(os.path.join(AUGMENTED_DIR, row["output"]), cv2.IMREAD_COLOR)

        before = f_analysis(degraded)
        corrected, tags, status = process(degraded.copy())
        brisque, mean, chroma, reason = check(corrected)

        labels = predict(tags)
        bucket = outcome(truth, labels, status == "ok")
        records.append({
            "output": row["output"], "truth": truth, "intensity": row["intensity"],
            "predicted": labels[0], "tags": "|".join(labels),
            "detected": int(detected(truth, labels)), "corrected": int(status == "ok"),
            "status": status, "outcome": bucket,
            "brisque_before": f"{before:.1f}", "brisque_after": f"{brisque:.1f}",
            "mean_before": row["mean_after"], "mean_after": f"{mean:.1f}",
            "chroma_before": row["chroma_after"], "chroma_after": f"{chroma:.2f}",
            "blocked_on": reason.split()[0] if reason else "-", "fail_reason": reason,
        })

        folder = os.path.join(run_dir, bucket, truth)
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, row["output"]), np.hstack([
            panel(source, "1. source", f"brisque {f_analysis(source):.0f}"),
            panel(degraded, f"2. degradee : {truth} x{float(row['intensity']):.2f}",
                  f"brisque {before:.0f}  mean {float(row['mean_after']):.0f}"
                  f"  chroma {float(row['chroma_after']):.1f}"),
            panel(corrected, f"3. corrigee -> {'|'.join(labels)}",
                  reason or "dans toutes les tolerances"),
        ]))
        print(f"[{index:3}/{len(rows)}] {row['output']:32} {truth:15} -> {'|'.join(labels):22} {bucket}")

    with open(os.path.join(run_dir, "results.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    matrix = collections.Counter((r["truth"], r["predicted"]) for r in records)
    with open(os.path.join(run_dir, "confusion.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("truth",) + COLUMNS)
        for truth in DEFECTS:
            writer.writerow([truth] + [matrix[(truth, column)] for column in COLUMNS])

    summary = "\n".join(report(records, matrix))
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(summary + "\n")

    print("\n" + summary)
    print(f"\n{len(records)} images en {time.time() - started:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
