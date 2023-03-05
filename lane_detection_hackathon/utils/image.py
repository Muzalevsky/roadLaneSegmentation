import numpy as np


def is_gray(img: np.array) -> bool:
    return len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)


def overlay(img: np.array, mask: np.array) -> np.array:
    return np.where(mask != (0, 0, 0), mask, img)
