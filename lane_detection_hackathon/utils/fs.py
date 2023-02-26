import json

import cv2
import numpy as np
import yaml

from .image import is_gray


def read_image(fpath: str, gray_scale: bool = False) -> np.ndarray:
    flag = None
    if gray_scale:
        flag = cv2.IMREAD_GRAYSCALE

    img = cv2.imread(fpath, flags=flag)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def write_image(fpath: str, img: np.array):
    if not is_gray(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(fpath, img)


def read_yaml(fpath: str) -> dict:
    with open(fpath) as fp:
        data = yaml.safe_load(fp)
    return data


def save_yaml(fpath: str, data: dict):
    with open(fpath, "w") as fp:
        yaml.safe_dump(data, fp)


def read_json(fpath: str) -> dict:
    with open(fpath) as fp:
        data = json.load(fp)
    return data[0]


def save_json(fpath: str, data: dict):
    with open(fpath, "w") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)
