from PIL import Image
import cv2
import numpy as np

def exp_analysis(img_arr):
    img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    return img_gray.mean()

def exp_correction(mean):

    pass