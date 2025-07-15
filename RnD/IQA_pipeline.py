import os
from exporuse_detection_test import computeMeanLuminosity
from brisque.brisque import BRISQUE
from PIL import Image
import numpy as np
from noise_detection_test import high_freq_mean

if __name__ == "__main__":
    # Calcul des moyennes
    results = {}
    for d, r, f in os.walk("../img/abberations/img1"):
        for filename in f:
            name = filename.replace(".png","_1")
            results[name] = high_freq_mean(os.path.join(d, filename))
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}

    for row in sorted_results:
        print(f"Frequency mean {row}: {sorted_results[row]}")
    results = {}
    print('')

    for d, r, f in os.walk("../img/abberations/img2"):
        for filename in f:
            name = filename.replace(".png", "_2")
            results[name] = high_freq_mean(os.path.join(d, filename))

    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}
    for row in sorted_results:
        print(f"Frequency mean {row}: {sorted_results[row]}")