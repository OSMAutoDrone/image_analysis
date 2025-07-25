import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os

def computeMeanLuminosity(image_path):
    img = Image.open(image_path)
    img_arr = np.asarray(img)
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    return img_gray.mean()


if __name__ == '__main__':
    # Calcul des moyennes
    results = {}
    for d, r, f in os.walk("../img/abberations/img1"):
        for filename in f:
            name = filename.replace(".png","_1")
            results[name] = computeMeanLuminosity(os.path.join(d, filename))
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