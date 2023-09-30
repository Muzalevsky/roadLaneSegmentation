from typing import NewType

import numpy as np


class Dict(dict):
    def keys(self) -> list:
        return list(super().keys())

    def values(self) -> list:
        return list(super().values())


Image = NewType("Image", np.ndarray)
ImageGray = NewType("ImageGray", Image)
ImageBinary = NewType("ImageBinary", Image)
ImageBinaryInverted = NewType("ImageBinaryInverted", Image)
ImageRGB = NewType("ImageRGB", Image)
ImageMask = NewType("ImageMaks", Image)
