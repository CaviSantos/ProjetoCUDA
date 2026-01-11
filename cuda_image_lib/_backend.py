import numpy as np

def rgb_to_gray_cuda(image: np.ndarray) -> np.ndarray:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)

def blur3x3_cuda(image: np.ndarray) -> np.ndarray:
    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]) / 9.0

    padded = np.pad(image, 1, mode="edge")
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+3, j:j+3]
            out[i, j] = np.sum(region * kernel)

    return out.astype(np.uint8)

def laplace3x3_cuda(image: np.ndarray) -> np.ndarray:
    kernel = np.array([
        [0,  1,  0],
        [1, -4,  1],
        [0,  1,  0]
    ])

    padded = np.pad(image, 1, mode="edge")
    out = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+3, j:j+3]
            out[i, j] = np.sum(region * kernel)

    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)
