import cv2
import numpy as np

# Thresholds separating the two frequency defects. Measured on the bench,
# p10-p90 per class:
#
#                 noise sigma      laplacian variance
#     blur       0.27 -  0.37         4 -    35
#     clean      0.62 -  3.21       111 -  1219
#     noise      7.81 - 16.18      1563 -  5704
#
# Noise is calculated using Immerkaer's mask, blurryness using the laplacian variance
NOISE_SIGMA = 5.0
BLUR_VARIANCE = 40

_SHARPEN_KERNEL = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], dtype=np.float32)
_BLUR_KERNEL = np.ones((3, 3), np.float32) / 9.0

# Immerkaer's mask. It cancels on any locally flat or linear patch, so image
# structure leaves almost no response and what survives is the noise.
_NOISE_KERNEL = np.array([[1, -2, 1],
                          [-2, 4, -2],
                          [1, -2, 1]], dtype=np.float32)


def f_noise(img_arr):
    """Return the noise standard deviation of a BGR image, in grey levels.

    The 6 divides out the mask's coefficient norm, whose squares sum to 36, and
    sqrt(pi/2) turns a mean absolute deviation back into a standard deviation.
    """
    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    response = np.abs(cv2.filter2D(gray, -1, _NOISE_KERNEL)).sum()
    return float(np.sqrt(np.pi / 2) * response / (6.0 * (width - 2) * (height - 2)))


def f_sharpness(img_arr):
    """Return the laplacian variance of a BGR image (higher = sharper).

    A second derivative responds to edges, and blurring attenuates them, so the
    spread of the response collapses as the image softens.
    """
    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def f_detect(img_arr):
    """Name the frequency defect: 'noise', 'blur' or None.

    Noise is tested first because it also drives the laplacian variance up: a
    noisy image would otherwise read as amply sharp and fall through.
    """
    if f_noise(img_arr) > NOISE_SIGMA:
        return "noise"
    if f_sharpness(img_arr) < BLUR_VARIANCE:
        return "blur"
    return None


def f_correction(img_arr, mode):
    """Correct blur ('blur' -> sharpen) or noise ('noise' -> 3x3 mean)."""
    kernel = _SHARPEN_KERNEL if mode == "blur" else _BLUR_KERNEL
    return cv2.filter2D(img_arr, -1, kernel)
