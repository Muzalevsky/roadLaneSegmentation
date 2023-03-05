import math

import cv2
import numpy as np


def is_gray(img: np.array) -> bool:
    return len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)


def overlay(img: np.array, mask: np.array) -> np.array:
    return np.where(mask != (0, 0, 0), mask, img)


def read_block(
    image: np.ndarray,
    xywh: tuple[int, int, int, int],
    pad_fill_value: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    x, y, w, h = np.array(xywh, dtype=int)
    x2, y2 = x + w, y + h

    im_h, im_w = image.shape[:2]
    block = image[y:y2, x:x2]

    if pad_fill_value is not None:
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
                value=pad_fill_value,
            )

    return block


def read_blocks(image: np.ndarray, cell_size_px: int) -> np.ndarray:
    w = math.ceil(image.shape[1] / cell_size_px)
    h = math.ceil(image.shape[0] / cell_size_px)

    blocks = []
    for row in range(h):
        for col in range(w):
            blocks.append(
                read_block(
                    image, (col * cell_size_px, row * cell_size_px, cell_size_px, cell_size_px)
                )
            )

    return blocks


def get_cells_geometry(w: int, h: int, cell_size_px: int) -> np.ndarray:
    cols = math.ceil(w / cell_size_px)
    rows = math.ceil(h / cell_size_px)

    cells_geometry = []
    for row in range(rows):
        for col in range(cols):
            cells_geometry.append(
                {
                    "index": row * cols + col,
                    "x": col * cell_size_px,
                    "y": row * cell_size_px,
                    "w": cell_size_px,
                    "h": cell_size_px,
                }
            )

    return cells_geometry


def glue_blocks(w: int, h: int, img_tiles: np.ndarray, gray_scale: bool = False) -> np.ndarray:
    block_w = img_tiles[0].shape[1]
    block_h = img_tiles[0].shape[0]

    cols = math.ceil(w / block_w)
    rows = math.ceil(h / block_h)

    final_shape = [rows * block_h, cols * block_w]
    if not gray_scale:
        final_shape.append(3)

    final = np.zeros((rows * block_h, cols * block_w, 3), np.uint8)
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            x = col * block_w
            y = row * block_h
            x2, y2 = x + block_w, y + block_h
            final[y:y2, x:x2] = img_tiles[index]

    cropped = final[0:h, 0:w]
    return cropped
