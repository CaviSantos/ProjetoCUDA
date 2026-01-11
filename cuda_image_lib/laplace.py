import numpy as np
from .validators import validate_gray
from ._backend import laplace3x3_cuda as _laplace3x3_cuda

def laplace3x3(image: np.ndarray) -> np.ndarray:
    validate_gray(image)
    return _laplace3x3_cuda(image)

