# noqa: flake8

from typing import Union

import logging
import numbers
import os

import cv2
import numpy as np

from . import patch_extractor as pe
from .utils import annot_parser_v2, math
from .utils.annot_mask import class_info_2_color_mapping

logger = logging.getLogger(__name__)


class ImageGridGenerator:
    def __init__(
        self,
        image: np.ndarray,
        cell_size: Union[tuple[int, int], int],
        step_size: Union[tuple[int, int], int] = None,
    ) -> None:
        self._image = image

        self._block_reader = pe.ImageBlockReader(self._image)

        if isinstance(cell_size, numbers.Number):
            cell_size = (cell_size, cell_size)

        if step_size is None:
            step_size = cell_size

        if isinstance(step_size, numbers.Number):
            step_size = (step_size, step_size)

        self._cell_sz = cell_size
        self._step_sz = step_size

        self._grid_size = self._compute_grid_size(
            src_sz=self.image_size, kernel_sz=self._cell_sz, stride=self._step_sz
        )
        # self._grid_size = np.ceil(self._extr.size / self._cell_sz)
        self._grid_size = self._grid_size.astype(int)
        logger.debug(f"Grid size: {self._grid_size}")

    @property
    def image_shape(self):
        return np.array(self._image.shape, dtype=int)

    @property
    def image_size(self):
        return np.array(self._image.shape[1::-1], dtype=int)

    @staticmethod
    def _compute_grid_size(src_sz, kernel_sz, stride):
        return np.ceil((src_sz - kernel_sz) / stride + 1)

    def __len__(self):
        # Equivalent of self._grid_size[0] * self._grid_size[1]
        return np.prod(self._grid_size)

    def _xy_to_offset(self, x, y):
        xy = np.array([x, y], dtype=np.float32)
        offset = xy * self._step_sz
        return offset.astype(int)

    # TODO - return patches instead of images
    def get_cell_2d(self, x: int, y: int, **kwargs) -> np.ndarray:
        """Obtain grid cell by 2d coordinates

        Parameters
        ----------
        x : int
            Column index (left to right)
        y : int
            Row index (top to bottom)

        Returns
        -------
        Tuple[patch_extractor.ImagePatch, patch_extractor.MaskPatch]
            Grid cell (img + mask)
        """

        offset = self._xy_to_offset(x, y)

        x, y = offset
        w, h = self._cell_sz

        im_block = self._block_reader.read_block((x, y, w, h))
        return im_block

    def get_cell(self, idx: int, **kwargs) -> pe.ImagePatch:
        """Obtain grid cell in row-wise scan order

        Parameters
        ----------
        idx : int
            Index of grid cell

        Returns
        -------
        Tuple[patch_extractor.ImagePatch, patch_extractor.MaskPatch]
            Grid cell (img + mask)
        """
        x, y = math.idx_2_xy(idx, self._grid_size[0])
        return self.get_cell_2d(x, y, **kwargs)


class RandomizedForegroundMultiplicationGenerator:
    def __init__(
        self,
        fpath: str,
        ann_fpath: str,
        ann_config: dict,
        multiplication_factor: int,
        cell_size: Union[tuple[int, int], int],
        target_resolution_um: float = None,
        offset_percent: float = 0.2,
        seed: int = 42,
    ) -> None:
        self._extr = pe.PatchExtractor(fpath)
        self._source_image_sz = self._extr.size
        self._mul_factor = multiplication_factor
        self._rng = np.random.default_rng(seed)
        self._offset_percent = offset_percent

        self._mask_extr = pe.AnnotMaskPatchExtractor(
            ann_fpath,
            image_size=self._extr.size,
            resolution=self._extr.resolution,
            ann_config=ann_config,
        )

        # Extract contours for bbox detection
        assert ann_fpath.lower().endswith(".xml")
        annots = annot_parser_v2.read_xml(ann_fpath)
        color_2_id_mapping = class_info_2_color_mapping(
            classes_info=ann_config["classes_info"], labels=ann_config["class_labels"]
        )

        self._foreground_segments = []

        # Filter only foreground classes
        for i in range(len(annots)):
            segment = annots[i]
            key_color = int(segment.ole_color)
            label_idx = color_2_id_mapping.get(key_color, -1)
            if label_idx < 0:
                continue

            self._foreground_segments.append(segment)

        self._source_resolution = self._extr.resolution

        if target_resolution_um is None:
            target_resolution_um = self._source_resolution
        else:
            target_resolution_um = np.array(
                [target_resolution_um, target_resolution_um], dtype=np.float32
            )

        self._target_resolution = target_resolution_um
        self._resize_ratio = self._target_resolution / self._source_resolution

        if isinstance(cell_size, numbers.Number):
            cell_size = (cell_size, cell_size)

        self._target_cell_sz = np.array(cell_size, dtype=int)

        # Should be int to fix size of cell
        # cell_sz - query from file cell
        # target_cell_sz - required result size (considering target resolution)
        self._cell_sz = self._target_cell_sz * self._resize_ratio
        self._cell_sz = self._cell_sz.astype(int)

    def __len__(self):
        # Equivalent of self._grid_size[0] * self._grid_size[1]
        return int(self._mul_factor * len(self._foreground_segments))

    def _get_cell_segment_coordinates(self, segment) -> pe.ImagePatch:
        contour = np.array(segment.polygon, dtype=int)

        bbox = cv2.boundingRect(contour)
        bb_x, bb_y, bb_w, bb_h = bbox

        offset_x = int(self._offset_percent * bb_w)
        offset_y = int(self._offset_percent * bb_h)

        x_limits = (bb_x - offset_x, bb_x - (self._cell_sz[0] - bb_w) + offset_x)
        y_limits = (bb_y - offset_y, bb_y - (self._cell_sz[1] - bb_h) + offset_y)

        x = self._rng.integers(low=min(*x_limits), high=max(*x_limits))
        y = self._rng.integers(low=min(*y_limits), high=max(*y_limits))

        return x, y

    def get_cell_img_ann(self, idx: int, **kwargs) -> tuple[pe.ImagePatch, pe.MaskPatch]:
        seg_idx = idx // self._mul_factor

        x, y = self._get_cell_segment_coordinates(self._foreground_segments[seg_idx])

        patch = self._extr.get_patch_by_offset(offset=(x, y), size=self._cell_sz)

        if not kwargs.get("skip_resize", False) and np.any(
            self._target_resolution != self._source_resolution
        ):
            patch.set_resolution(self._target_resolution, self._target_cell_sz)

        mask_patch = self._mask_extr.get_patch_by_offset(offset=(x, y), size=self._cell_sz)

        if not kwargs.get("skip_resize", False) and np.any(
            self._target_resolution != self._source_resolution
        ):
            mask_patch.set_resolution(self._target_resolution, self._target_cell_sz)

        return patch, mask_patch


class RegularGridGenerator:
    def __init__(
        self,
        fpath: str,
        cell_size: Union[tuple[int, int], int],
        step_size: Union[tuple[int, int], int] = None,
        target_resolution_um: float = None,
        ann_fpath: str = None,
        ann_config: dict = None,
    ) -> None:
        self._extr = pe.PatchExtractor(fpath)
        self._source_image_sz = self._extr.size

        self._mask_extr = None
        if ann_fpath is not None and os.path.exists(ann_fpath):
            assert ann_config is not None

            self._mask_extr = pe.AnnotMaskPatchExtractor(
                ann_fpath,
                image_size=self._extr.size,
                resolution=self._extr.resolution,
                ann_config=ann_config,
            )

        self._source_resolution = self._extr.resolution

        if target_resolution_um is None:
            target_resolution_um = self._source_resolution
        else:
            target_resolution_um = np.array(
                [target_resolution_um, target_resolution_um], dtype=np.float32
            )

        self._target_resolution = target_resolution_um
        self._resize_ratio = self._target_resolution / self._source_resolution

        if isinstance(cell_size, numbers.Number):
            cell_size = (cell_size, cell_size)

        if step_size is None:
            step_size = cell_size

        if isinstance(step_size, numbers.Number):
            step_size = (step_size, step_size)

        self._target_cell_sz = np.array(cell_size, dtype=int)
        self._target_step_sz = np.array(step_size, dtype=int)

        # Should be int to fix size of cell
        self._cell_sz = self._target_cell_sz * self._resize_ratio
        self._cell_sz = self._cell_sz.astype(int)

        # This one can be float to compute correct cell offsets
        self._step_sz = self._target_step_sz * self._resize_ratio

        logger.debug(f"Img size: {self._extr.size}")
        logger.debug(f"Target cell size: {self._target_cell_sz}")
        logger.debug(f"Read cell size: {self._cell_sz}")
        logger.debug(f"Target step size: {self._target_step_sz}")
        logger.debug(f"Read step size: {self._step_sz}")

        self._grid_size = self._compute_grid_size(
            src_sz=self._source_image_sz, kernel_sz=self._cell_sz, stride=self._step_sz
        )
        # self._grid_size = np.ceil(self._extr.size / self._cell_sz)
        self._grid_size = self._grid_size.astype(int)
        logger.debug(f"Grid size: {self._grid_size}")

    @property
    def image_size(self):
        return np.array(self._extr.size, dtype=int)

    @property
    def source_resolution(self):
        return self._extr._source_resolution

    @staticmethod
    def _compute_grid_size(src_sz, kernel_sz, stride):
        return np.ceil((src_sz - kernel_sz) / stride + 1)

    def __len__(self):
        # Equivalent of self._grid_size[0] * self._grid_size[1]
        return np.prod(self._grid_size)

    def _xy_to_offset(self, x, y):
        xy = np.array([x, y], dtype=np.float32)
        offset = xy * self._step_sz
        return offset.astype(int)

    def get_cell_img_2d(self, x: int, y: int, **kwargs) -> pe.ImagePatch:
        """Obtain grid cell by 2d coordinates

        Parameters
        ----------
        x : int
            Column index (left to right)
        y : int
            Row index (top to bottom)

        Returns
        -------
        Tuple[patch_extractor.ImagePatch, patch_extractor.MaskPatch]
            Grid cell (img + mask)
        """

        offset = self._xy_to_offset(x, y)

        patch = self._extr.get_patch_by_offset(offset=offset, size=self._cell_sz)

        if not kwargs.get("skip_resize", False) and np.any(
            self._target_resolution != self._source_resolution
        ):
            patch.set_resolution(self._target_resolution, self._target_cell_sz)

        return patch

    def get_cell_ann_2d(self, x: int, y: int, **kwargs) -> pe.MaskPatch:
        offset = self._xy_to_offset(x, y)

        if self._mask_extr is None:
            raise ValueError("")

        mask_patch = self._mask_extr.get_patch_by_offset(offset=offset, size=self._cell_sz)

        if not kwargs.get("skip_resize", False) and np.any(
            self._target_resolution != self._source_resolution
        ):
            mask_patch.set_resolution(self._target_resolution, self._target_cell_sz)

        return mask_patch

    def get_cell_img(self, idx: int, **kwargs) -> pe.ImagePatch:
        """Obtain grid cell in row-wise scan order

        Parameters
        ----------
        idx : int
            Index of grid cell

        Returns
        -------
        Tuple[patch_extractor.ImagePatch, patch_extractor.MaskPatch]
            Grid cell (img + mask)
        """
        x, y = math.idx_2_xy(idx, self._grid_size[0])
        return self.get_cell_img_2d(x, y, **kwargs)

    def get_cell_ann(self, idx: int, **kwargs) -> pe.MaskPatch:
        x, y = math.idx_2_xy(idx, self._grid_size[0])
        return self.get_cell_ann_2d(x, y, **kwargs)
