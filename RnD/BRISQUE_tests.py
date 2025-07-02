import numpy as np
import matplotlib.pyplot as plt
import cv2
from brisque.brisque import BRISQUE
from PIL import Image

img = Image.open('E:/Perdoudes/image_analysis/img/abberations/img1/gaussianBlur.png')
ndarray = np.asarray(img)

obj = BRISQUE(url=False)
score = obj.score(img=ndarray)
print(score)