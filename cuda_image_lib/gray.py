import numpy as np
from .validators import validate_rgb
from ._backend import rgb_to_gray_cuda as _rgb_to_gray_cuda

def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    validate_rgb(image)
    return _rgb_to_gray_cuda(image)

