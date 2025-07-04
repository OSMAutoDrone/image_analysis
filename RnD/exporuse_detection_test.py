import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def computeMeanLuminosity(image_path):
    img = Image.open(image_path)
    img_arr = np.asarray(img)
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    return img_gray.mean()


if __name__ == '__main__':

    print(f"mean image base: {computeMeanLuminosity("../img/abberations/img1/base.png")}")
    print(f"mean image overEx: {computeMeanLuminosity("../img/abberations/img1/overexposed.png")}")
    print(f"mean image underEx: {computeMeanLuminosity("../img/abberations/img1/underexposed.png")}")
    print(f"mean image noisy: {computeMeanLuminosity("../img/abberations/img1/gaussianNoise.png")}")
    print("")

    print(f"mean image2 base: {computeMeanLuminosity("../img/abberations/img2/base.png")}")
    print(f"mean image2 overEx: {computeMeanLuminosity("../img/abberations/img2/overexposed.png")}")
    print(f"mean image2 underEx: {computeMeanLuminosity("../img/abberations/img2/underexposed.png")}")
    print(f"mean image2 noisy: {computeMeanLuminosity("../img/abberations/img2/gaussianNoise.png")}")
