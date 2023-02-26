import cv2
import numpy as np


class ImageBlockReader:
    def __init__(self, image, pad_fill_value: int = None):
        self._image = image
        self._pad_fill_value = pad_fill_value
        self._im_h, self._im_w = self._image.shape[:2]

    def read_block(self, xywh: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = np.array(xywh, dtype=int)
        x2, y2 = x + w, y + h

        block = self._image[y:y2, x:x2]

        if self._pad_fill_value is not None:
            top_pad = np.clip(-y, a_min=0, a_max=None)
            left_pad = np.clip(-x, a_min=0, a_max=None)
            right_pad = np.clip(x2 - self._im_w, a_min=0, a_max=None)
            bottom_pad = np.clip(y2 - self._im_h, a_min=0, a_max=None)

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
