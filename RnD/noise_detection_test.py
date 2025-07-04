import cv2
import numpy as np
import os

# Charger les images en niveaux de gris
img1 = cv2.imread("../img/abberations/img1/base.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("../img/abberations/img1/gaussianNoise.png", cv2.IMREAD_GRAYSCALE)


# Fonction pour extraire la moyenne des hautes fréquences
def high_freq_mean(img_path, cutoff=30):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    # Masque circulaire pour ne garder que les hautes fréquences
    h, w = img.shape
    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    mask = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2) > cutoff

    return np.mean(mag)


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


