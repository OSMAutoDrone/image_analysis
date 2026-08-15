from PIL import Image
import numpy as np
import cv2
from exporuse_detection_test import *
from brisque.brisque import BRISQUE

if __name__ == '__main__':
    img =cv2.imread("../img/abberations/img1/gaussianBlur.png", flags=cv2.IMREAD_COLOR)
    ndarray = np.asarray(img)
    obj = BRISQUE(url=False)
    score = obj.score(img=ndarray)
    print(f"niveau initial (sharp){score}")
    sharpening_filter = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, sharpening_filter)
    sharpened = cv2.filter2D(sharpened, -1, sharpening_filter)
    score = obj.score(img=sharpened)
    print(f"niveau corrigé (sharp){score}")
    cv2.imshow('Defect', img)
    cv2.imshow('Corrected', sharpened)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img = cv2.imread("../img/abberations/img1/gaussianNoise.png", flags=cv2.IMREAD_COLOR)
    blurring_filter = np.array(([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]]))
    # blurring_filter = blurring_filter/9
    blurred = cv2.morphologyEx(img, cv2.MORPH_OPEN, blurring_filter)
    cv2.imshow('Defect', img)
    cv2.imshow('Corrected', blurred)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img = cv2.imread("../img/abberations/img1/parking.jpeg", flags=cv2.IMREAD_COLOR)
    img_arr = np.asarray(img)
    level = computeMeanLuminosity(img)
    print(f"niveau initial (lum){level}")
    correction = (255-level)/127.5
    sharpening_filter = np.array([[0, 0, 0],
                                  [0, correction, 0],
                                  [0, 0, 0]])
    sharpened = cv2.filter2D(img, -1, sharpening_filter)
    print(f"niveau corrigé (lum) {computeMeanLuminosity(sharpened)}")
    cv2.imshow('Defect', img)
    cv2.imshow('Corrected', sharpened)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img = cv2.imread("../img/abberations/img1/overexposed.png", flags=cv2.IMREAD_COLOR)
    img_arr = np.asarray(img)
    level = computeMeanLuminosity(img)
    correction = (255-level)/127.5
    sharpening_filter = np.array([[0, 0, 0],
                                  [0, correction, 0],
                                  [0, 0, 0]])
    sharpened = cv2.filter2D(img, -1, sharpening_filter)
    sharpened = cv2.filter2D(sharpened, -1, sharpening_filter)
    cv2.imshow('Defect', img)
    cv2.imshow('Corrected', sharpened)
    cv2.waitKey()
    cv2.destroyAllWindows()