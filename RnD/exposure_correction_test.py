import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

req = Image.open("C:/Users/ti-da\OneDrive\Documents\PMC\image_analysis\img/abberations/img1/overexposed.png")
img = np.asarray(req)


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_th = cv2.adaptiveThreshold(
    img_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    1
)


plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.xticks([]),plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(img_th)
plt.title('Adaptive Gaussian Thresholding')
plt.xticks([]),plt.yticks([])

plt.show()