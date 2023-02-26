# noqa: flake8

from typing import Union

import logging
import os

import cv2
import numpy as np
import slideio
from shapely import geometry

from .utils import annot_mask, math

logger = logging.getLogger(__name__)


class ImagePatch:
    def __init__(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        resolution: np.ndarray = None,
        resize_interp: int = cv2.INTER_LINEAR,
    ):
        self._image = image
        self._resolution = resolution
        self._resize_interp = resize_interp
        self._position = np.array(position, dtype=np.float32)

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def info(self) -> dict:
        return dict(
            resolution=self._resolution,
            position=self._position,
        )

    @property
    def position(self) -> tuple[int, int]:
        return self._position

    @property
    def shape(self) -> tuple[int, int]:
        return self._image.shape

    @property
    def size(self) -> tuple[int, int]:
        return self.shape[1::-1]

    @property
    def resolution(self) -> tuple[float, float]:
        return self._resolution

    def as_xywh(self) -> tuple:
        return (*self._position, *self.size)

    def as_shapely_box(self) -> geometry.Polygon:
        x, y = self._position
        h, w = self.shape[:2]
        return geometry.box(x, y, x + w, y + h)

    def overlap_ratio(self, polygon: geometry.Polygon) -> float:
        self_shape = self.as_shapely_box()
        ratio = self_shape.intersection(polygon).area / self_shape.area
        return ratio

    def __repr__(self):
        return f"Patch at {self._position}, size {self.shape[:2]}, resolution {self._resolution}"

    def set_resolution(
        self, resolution: float, size: Union[tuple[int, int], None] = None
    ) -> np.ndarray:
        resize_ratio = resolution / self._resolution
        if size is None:
            src_size = np.array(self.size, dtype=int)
            size = src_size / resize_ratio
            size = size.astype(int)

        # Update coordinates
        self._position /= resize_ratio

        self._resolution = np.full_like(self._resolution, fill_value=resolution)
        self._image = cv2.resize(self._image, dsize=size, interpolation=self._resize_interp)


class MaskPatch(ImagePatch):
    def __init__(self, **kwargs):
        kwargs.setdefault("resize_interp", cv2.INTER_NEAREST)
        super().__init__(**kwargs)


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


class PatchExtractor:
    def __init__(self, fpath):
        if not os.path.exists(fpath):
            raise ValueError(f"File not exists: {fpath}")

        assert fpath.lower().endswith(".svs"), "Filepath have to have `.svs` extention"

        self._fpath = fpath

        self._slide = slideio.open_slide(path=self._fpath, driver="SVS")
        self._scene = self._slide.get_scene(0)
        self._size = np.array(self._scene.size, dtype=np.float32)

        self._source_resolution = np.array(self._scene.resolution, dtype=np.float32)
        # Convert to microns [um]
        self._source_resolution *= 1e6

        logger.debug(f"Source scene resolution: {self._source_resolution} [um]")

    @property
    def resolution(self):
        return self._source_resolution

    def get_patch_by_offset(self, offset: tuple[int, int], size: tuple[int, int]) -> ImagePatch:
        x, y = offset
        w, h = size

        logger.debug(f"Extract by coordinates {offset} with size {size}")
        block = self._scene.read_block((x, y, w, h))
        patch = ImagePatch(image=block, position=(x, y), resolution=self.resolution)
        return patch

    def get_patch_by_center(self, center: tuple[int, int], size: tuple[int, int]) -> ImagePatch:
        center = np.array(center, dtype=np.float32)
        offset = center - np.array(size, dtype=np.float32) / 2

        patch = self.get_patch_by_offset(offset.astype(np.int32), size)
        return patch

    @property
    def size(self):
        return self._size


class AnnotMaskPatchExtractor:
    def __init__(
        self, fpath: str, image_size: tuple[int, int], resolution: float, ann_config: dict
    ) -> None:
        self._fpath = fpath
        self._image_size = image_size
        self._resolution = resolution
        self._ann_config = ann_config

        self._ann_mask_gen = annot_mask.AnnotMask(
            fpath=self._fpath, image_size=image_size, config=self._ann_config
        )
        self._global_mask = self._ann_mask_gen.full_mask

    def get_patch_by_offset(self, offset: tuple[int, int], size: tuple[int, int]) -> MaskPatch:
        # Negative offset not supported yet
        assert np.all(np.array(offset) >= 0)
        logger.debug(f"Get patch with parameters: {offset}, {size}")

        x, y = offset
        w, h = size

        block = self._ann_mask_gen.read_block((x, y, w, h))

        return MaskPatch(image=block, position=(x, y), resolution=self._resolution)


# NOTE - not used
class PatchExtractorMetric(PatchExtractor):
    def __init__(self, fpath):
        super().__init__(fpath)

        self._metric_size = self._size * self.resolution

    def get_patch_by_offset(
        self, offset: tuple[float, float], size: tuple[float, float]
    ) -> ImagePatch:
        """Extract patch from image by top-left offset coordinates

        Args:
            offset (Tuple[float, float]): Top-left offset coordinates [microns]
            size (Tuple[float, float]): Patch size [microns]

        Returns:
            ImagePatch: Patch image
        """
        offset_px = math.microns_2_px(offset, self.resolution)
        size_px = math.microns_2_px(size, self.resolution)

        patch = super().get_patch_by_offset(offset_px.astype(np.int32), size_px.astype(np.int32))
        return patch

    def get_patch_by_center(
        self, center: tuple[float, float], size: tuple[float, float]
    ) -> ImagePatch:
        """Extract patch from image by center coordinates

        Args:
            center (Tuple[float, float]): Center coordinates [microns]
            size (Tuple[float, float]): Patch size [microns]

        Returns:
            ImagePatch: Patch image
        """
        center_px = math.microns_2_px(center, self.resolution)
        size_px = math.microns_2_px(size, self.resolution)

        patch = super().get_patch_by_center(center_px.astype(np.int32), size_px.astype(np.int32))
        return patch

    @property
    def size_m(self):
        return self._metric_size
