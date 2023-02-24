import logging

import numpy as np

from .utils.types import Dict, ImageGray, ImageRGB


def get_mask_map():
    return Dict(
        {
            "SYD": (60, 15, 67),  # solid yellow dividing
            "BWG": (142, 35, 8),  # broken white guiding
            "SWD": (180, 173, 43),  # solid white dividing
            "SWS": (0, 0, 192),  # solid white stopping
            "CWYZ": (153, 102, 153)  # zebra
            # "AWR": (35, 136, 226),      # arrow white right turn
            # "ALW": (180, 109, 91),      # arrow white left turn
            # "AWTL": (160, 168, 234),    # arrow white thru & left turn
        }
    )


class MaskProcessor:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._map = get_mask_map()

    def to_label_mask(self, mask_img: ImageRGB, label_map: dict) -> ImageGray:
        label_mask = np.zeros(mask_img.shape[:2])
        for label_name, label_id in label_map.items():
            label_color = self._map[label_name]
            yy, xx, _zz = np.where(mask_img == label_color)
            label_mask[yy, xx] = label_id

        return label_mask

    def to_ohe_mask(self, mask_img: ImageGray, label_map: Dict):
        one_hot_mask = np.zeros(*mask_img.shape[:2], len(label_map))
        for i, label_id in enumerate(label_map.values()):
            one_hot_mask[:, :, i][mask_img == label_id] = 1

        return one_hot_mask
