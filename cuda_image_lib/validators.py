import numpy as np

def validate_rgb(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("A imagem precisa ser um numpy array.")
    if image.dtype != np.uint8:
        raise TypeError("A imagem precisa ser no formato uint8.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("A imagem RGB deve ter formato (H, W, 3).")


def validate_gray(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("A imagem precisa ser um numpy array.")
    if image.dtype != np.uint8:
        raise TypeError("A imagem precisa ser uint8.")
    if image.ndim != 2:
        raise TypeError("A imagem cinza deve ter formato (H, W)")