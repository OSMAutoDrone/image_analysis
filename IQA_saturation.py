import cv2
import numpy as np

# Mean LAB chroma range considered correctly saturated.
SAT_LOW = 6.0
SAT_HIGH = 30.0

# Below this chroma the shot is a genuine greyscale image rather than a
# washed-out one: scaling its chroma up would never converge, so it is left
# alone. Measured floor of a x0.30 desaturation is 1.56, which stays above.
SAT_GREY = 1.0

_SAT_TARGET = (SAT_LOW + SAT_HIGH) / 2.0

# A near-neutral image would otherwise demand an enormous gain and would only
# amplify chroma noise; clipped LAB values also make large gains ineffective.
_SAT_FACTOR_RANGE = (0.4, 2.5)


def sat_analysis(img_arr):
    """Return the mean LAB chroma of a BGR image.

    LAB chroma is preferred over the HSV saturation channel because the latter
    is a ratio: it blows up on dark pixels and collapses on near-white ones, so
    it tracks exposure as much as colourfulness. On this dataset a dark image
    measured S=198 in HSV for a perceptual chroma half that of a genuinely
    vivid one.
    """
    lab = cv2.cvtColor(img_arr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0).mean())


def sat_scale(img_arr, factor):
    """Scale the a*/b* chroma of a BGR image around the neutral axis.

    Shared with the aberration generator so degradation and correction apply
    the very same transform in opposite directions.
    """
    lab = cv2.cvtColor(img_arr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1:] = (lab[..., 1:] - 128.0) * factor + 128.0
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def sat_correction(img_arr, chroma):
    """Pull the saturation toward the middle of the target range.

    The factor is dynamic: > 1 revives a washed-out image, < 1 tames an
    over-saturated one. Unlike exp_correction it aims straight at the target,
    so a single pass converges when the factor is not clamped.
    """
    factor = float(np.clip(_SAT_TARGET / max(chroma, 1e-6), *_SAT_FACTOR_RANGE))
    return sat_scale(img_arr, factor)
