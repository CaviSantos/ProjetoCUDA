import numpy as np
from cuda_image_lib import rgb_to_gray

img = np.zeros((100, 100, 3), dtype=np.uint8)

try:
    rgb_to_gray(img)
except NotImplementedError as e:
    print("API OK:", e)
