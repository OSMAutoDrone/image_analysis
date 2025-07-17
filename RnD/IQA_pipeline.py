import os
from exporuse_detection_test import computeMeanLuminosity
from brisque.brisque import BRISQUE
from PIL import Image
import numpy as np
from noise_detection_test import high_freq_mean

if __name__ == "__main__":
    results = {}
    for d, r, f in os.walk(f'../img/random/'):
        for filename in f:
            img = Image.open(os.path.join(d, filename))
            name = filename.replace(".jpg", f"")
            Exposition_score = computeMeanLuminosity(os.path.join(d, filename))
            if Exposition_score > 160:
                results[name] = "over-exposed"
                continue
            if Exposition_score < 70:
                results[name] = "under-exposed"
                continue
            ndarray = np.asarray(img)
            obj = BRISQUE(url=False)
            score = obj.score(img=ndarray)
            if score > 30:
                results[name] = "blurry or noisy"
                continue


    for row in results:
        print(f"{row}: {results[row]}")