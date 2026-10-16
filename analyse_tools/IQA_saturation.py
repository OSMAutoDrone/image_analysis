import cv2
import numpy as np

# Mean LAB chroma range considered correctly saturated.
SAT_LOW = 6.0
SAT_HIGH = 30.0

SAT_GREY = 1.0

_SAT_TARGET = (SAT_LOW + SAT_HIGH) / 2.0

# clipped max correction factors
_SAT_FACTOR_RANGE = (0.4, 2.5)


def sat_analysis(img_arr):
    """Return the mean LAB chroma of a BGR image.
    """
    lab = cv2.cvtColor(img_arr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0).mean())


def sat_scale(img_arr, factor):
    """Scale the a*/b* chroma of a BGR image around the neutral axis.
    """
    lab = cv2.cvtColor(img_arr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1:] = (lab[..., 1:] - 128.0) * factor + 128.0
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def sat_correction(img_arr, chroma):
    """Pull the saturation toward the middle of the target range.
    """
    factor = float(np.clip(_SAT_TARGET / max(chroma, 1e-6), *_SAT_FACTOR_RANGE))
    return sat_scale(img_arr, factor)
