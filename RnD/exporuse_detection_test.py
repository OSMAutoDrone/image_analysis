import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

def computeMeanLuminosity(img_arr):
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    return img_gray.mean()

def correction(image_path, level):
    img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    correction = (255 - level) / 127.5
    sharpening_filter = np.array([[0, 0, 0],
                                  [0, correction, 0],
                                  [0, 0, 0]])
    sharpened = cv2.filter2D(img, -1, sharpening_filter)
    sharpened = cv2.filter2D(sharpened, -1, sharpening_filter)
    cv2.imshow('Base', img)
    cv2.imshow('corrected', sharpened)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Calcul des moyennes
    results = {}
    for d, r, f in os.walk("../img/abberations/img1"):
        for filename in f:
            name = filename.replace(".png","_1")
            results[name] = computeMeanLuminosity(os.path.join(d, filename))
            correction(os.path.join(d, filename), results[name])
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

    for row in sorted_results:
        print(f"Mean exposure {row}: {sorted_results[row]}")
    results = {}
    print('')

    for d, r, f in os.walk("../img/abberations/img2"):
        for filename in f:
            name = filename.replace(".png", "_2")
            results[name] = computeMeanLuminosity(os.path.join(d, filename))

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}
    for row in sorted_results:
        print(f"Mean exposure {row}: {sorted_results[row]}")