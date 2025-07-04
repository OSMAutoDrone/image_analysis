import numpy as np
import matplotlib.pyplot as plt
import cv2
from brisque.brisque import BRISQUE
from PIL import Image
import os

for i in [1, 2]:
    results = {}
    for d, r, f in os.walk(f'../img/abberations/img{i}/'):
        for filename in f:
            img = Image.open(os.path.join(d, filename))
            ndarray = np.asarray(img)

            obj = BRISQUE(url=False)
            score = obj.score(img=ndarray)
            name = filename.replace(".png", f"_img{i}")
            # print(f"{name} : {score}")
            plt.figure()
            plt.imshow(img)
            plt.title(f"{name}: {score}")
            results[name] = score
    plt.show()
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}
    for row in sorted_results:
        print(f"{row}: {sorted_results[row]}")
    print('')

