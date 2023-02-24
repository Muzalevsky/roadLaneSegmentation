import cv2
import numpy as np


def read_image(fpath: str) -> np.ndarray:
    img = cv2.imread(fpath)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_mask(fpath: str) -> np.ndarray:
    """Read grayscale index-based mask."""
    mask = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    return mask
