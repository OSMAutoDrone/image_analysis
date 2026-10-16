import cv2
import numpy as np

# Mean luminance range (0-255) considered correctly exposed.
EXP_LOW = 70
EXP_HIGH = 160

# Middle of that range: the target for correction
EXP_TARGET = (EXP_LOW + EXP_HIGH) / 2.0

# Gain bounds, so a near-black image does not get a huge factor that would only
# amplify its noise.
EXP_FACTOR_RANGE = (0.2, 5.0)


def exp_analysis(img_arr):
    """Return the mean luminance (0-255) of a BGR image."""
    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    return gray.mean()


def exp_correction(img_arr, mean):
    """Scale the image so its mean luminance lands on EXP_TARGET.
    """
    factor = float(np.clip(EXP_TARGET / max(mean, 1e-6), *EXP_FACTOR_RANGE))
    return cv2.convertScaleAbs(img_arr, alpha=factor)
