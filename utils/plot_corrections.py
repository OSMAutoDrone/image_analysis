"""Plot how each metric moves from its degraded value to its corrected one.

One line per image, drawn across the four gates the pipeline enforces, with the
tolerance band shaded. Reads the results.csv of a run produced by
evaluate_pipeline and writes a PNG next to it.
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyse_tools.IQA_frequency import NOISE_SIGMA, BLUR_VARIANCE
from analyse_tools.IQA_exposition import EXP_LOW, EXP_HIGH
from analyse_tools.IQA_saturation import SAT_LOW, SAT_HIGH

RUNS_DIR = "runs"
FIGURE = "corrections.png"

PANELS = (
    ("noise", "Bruit  (sigma, niveaux de gris)", (0.0, NOISE_SIGMA), "log"),
    ("sharp", "Nettete  (variance du laplacien)", (BLUR_VARIANCE, None), "log"),
    ("mean", "Exposition  (luminance moyenne)", (EXP_LOW, EXP_HIGH), "linear"),
    ("chroma", "Saturation  (chroma LAB)", (SAT_LOW, SAT_HIGH), "linear"),
)

OUT_OF_BAND = "#C0392B"
IN_BAND = "#9AA7B2"
BAND = "#3F7D6E"


def latest_run():
    """Return the most recent run folder, or the one given on the command line."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    runs = sorted(d for d in os.listdir(RUNS_DIR) if d.startswith("eval_"))
    return os.path.join(RUNS_DIR, runs[-1])


def draw(axis, rows, column, title, band, scale):
    """Draw one before/after panel, one line per image."""
    low, high = band
    pairs = [(float(r[f"{column}_before"]), float(r[f"{column}_after"])) for r in rows]
    values = [v for pair in pairs for v in pair]

    axis.set_yscale(scale)
    if scale == "log":
        axis.set_ylim(min(values) * 0.7, max(values) * 1.4)
    else:
        margin = (max(values) - min(values)) * 0.06
        axis.set_ylim(min(values) - margin, max(values) + margin)
    bottom, top = axis.get_ylim()

    axis.axhspan(low if low is not None else bottom, high if high is not None else top,
                 color=BAND, alpha=0.11, zorder=0)
    for line in (low, high):
        if line:
            axis.axhline(line, color=BAND, lw=1.1, ls="--", zorder=1)

    outside = 0
    for before, after in pairs:
        bad = (low is not None and before < low) or (high is not None and before > high)
        outside += bad
        axis.plot([0, 1], [before, after], lw=0.8, alpha=0.45 if bad else 0.16,
                  color=OUT_OF_BAND if bad else IN_BAND, zorder=3 if bad else 2)

    axis.set_xlim(-0.12, 1.12)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["avant", "apres"], fontsize=11)
    axis.set_title(title, fontsize=12, pad=10, loc="left")
    axis.tick_params(labelsize=9)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

    still = sum(
        ((low is not None and float(r[f"{column}_after"]) < low)
         or (high is not None and float(r[f"{column}_after"]) > high))
        for r in rows)
    axis.text(0.5, -0.17, f"hors tolerance : {outside} avant  ->  {still} apres",
              transform=axis.transAxes, ha="center", fontsize=10, color=OUT_OF_BAND)


def main():
    run_dir = latest_run()
    with open(os.path.join(run_dir, "results.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    figure, axes = plt.subplots(1, 4, figsize=(16, 5.2))
    for axis, (column, title, band, scale) in zip(axes, PANELS):
        draw(axis, rows, column, title, band, scale)

    figure.suptitle(f"Convergence des {len(rows)} images vers les tolerances du pipeline",
                    fontsize=15, y=0.99, x=0.065, ha="left")
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_path = os.path.join(run_dir, FIGURE)
    figure.savefig(out_path, dpi=160, facecolor="white")
    print(f"{out_path}")


if __name__ == "__main__":
    main()
