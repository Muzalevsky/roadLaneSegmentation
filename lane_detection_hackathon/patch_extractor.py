import math

import cv2
import numpy as np


class ImageBlockReader:
    def __init__(self, pad_fill_value: int = None):
        self._pad_fill_value = pad_fill_value

    def read_block(self, image: np.ndarray, xywh: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = np.array(xywh, dtype=int)
        x2, y2 = x + w, y + h

        im_h, im_w = image.shape[:2]
        block = image[y:y2, x:x2]

        if self._pad_fill_value is not None:
            top_pad = np.clip(-y, a_min=0, a_max=None)
            left_pad = np.clip(-x, a_min=0, a_max=None)
            right_pad = np.clip(x2 - im_w, a_min=0, a_max=None)
            bottom_pad = np.clip(y2 - im_h, a_min=0, a_max=None)

            if any([top_pad, bottom_pad, left_pad, right_pad]):
                block = cv2.copyMakeBorder(
                    src=block,
                    top=top_pad,
                    bottom=bottom_pad,
                    left=left_pad,
                    right=right_pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value=self._pad_fill_value,
                )

        return block

    def read_blocks(self, image: np.ndarray, cell_size_px: int) -> np.ndarray:
        w = math.ceil(image.shape[1] / cell_size_px)
        h = math.ceil(image.shape[0] / cell_size_px)

        blocks = []
        for row in range(h):
            for col in range(w):
                blocks.append(
                    self.read_block(
                        image, (col * cell_size_px, row * cell_size_px, cell_size_px, cell_size_px)
                    )
                )

        return blocks
