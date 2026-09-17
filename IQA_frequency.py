import cv2
import numpy as np
from brisque.brisque import BRISQUE

# Above this BRISQUE score the image is considered blurred or noisy.
#
# Raised from the literature's 30 because BRISQUE is trained on natural
# photographs and scores aerial imagery far too harshly: on 40 pristine source
# images the median is 21.2 but the p90 reaches 40.8, and 28% already exceed 30
# with no defect at all. 42 sits just above that p90, so roughly nine pristine
# images out of ten pass. Provisional -- it rests on n=40 and deserves a proper
# statistical study of the BRISQUE response on this domain.
BRISQUE_THRESHOLD = 42

# Thresholds telling the two defects apart once BRISQUE has flagged one.
# Measured on the bench, p10-p90 per class:
#
#                 noise sigma      laplacian variance
#     blur       0.27 -  0.37         4 -    35
#     clean      0.62 -  3.21       111 -  1219
#     noise      7.81 - 16.18      1563 -  5704
#
# Noise separates with a wide margin. BLUR_VARIANCE is deliberately kept on the
# low side: the laplacian variance also falls with exposure, so 6 of the bench's
# underexposed images already dip under 40. Raising it to 60 catches one more
# blurred image at the cost of five more false positives.
NOISE_SIGMA = 5.0
BLUR_VARIANCE = 40

_SHARPEN_KERNEL = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], dtype=np.float32)
_BLUR_KERNEL = np.ones((3, 3), np.float32) / 9.0

# Immerkaer's mask. It cancels on any locally smooth or linear patch, so what
# is left of its response is the noise, and the mean absolute response scales
# with the noise standard deviation.
_NOISE_KERNEL = np.array([[1, -2, 1],
                          [-2, 4, -2],
                          [1, -2, 1]], dtype=np.float32)

_brisque = BRISQUE(url=False)


def f_analysis(img_arr):
    """Return the BRISQUE score (higher = more degraded)."""
    return _brisque.score(img=img_arr)


def f_detect(img_arr):
    """Name the defect BRISQUE flagged: 'noise', 'blur' or None.

    BRISQUE reports that an image is degraded, never how, so both corrections
    used to be tried in turn -- which sharpened noise before smoothing it and
    smoothed blur after sharpening it. These two spatial measures decide instead.

    Noise is tested first because it also drives the laplacian variance up: a
    noisy image would otherwise read as amply sharp and fall through.

    Returns None when neither signature fits, meaning BRISQUE dislikes the image
    for a reason this stage can neither name nor correct.
    """
    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    response = np.abs(cv2.filter2D(gray.astype(np.float32), -1, _NOISE_KERNEL)).sum()
    if np.sqrt(np.pi / 2) * response / (6.0 * (width - 2) * (height - 2)) > NOISE_SIGMA:
        return "noise"
    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_VARIANCE:
        return "blur"
    return None


def f_correction(img_arr, mode):
    """Correct blur ('blur' -> sharpen) or noise ('noise' -> 3x3 mean)."""
    kernel = _SHARPEN_KERNEL if mode == "blur" else _BLUR_KERNEL
    return cv2.filter2D(img_arr, -1, kernel)
