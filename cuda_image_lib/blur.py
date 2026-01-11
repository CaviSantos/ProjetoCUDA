import numpy as np
from .validators import validate_gray
from ._backend import blur3x3_cuda as _blur3x3_cuda

def blur3x3(image: np.ndarray) -> np.ndarray:
    validate_gray(image)
    return _blur3x3_cuda(image)