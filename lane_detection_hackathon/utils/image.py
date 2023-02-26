import numpy as np


def is_gray(img: np.array) -> bool:
    return len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
