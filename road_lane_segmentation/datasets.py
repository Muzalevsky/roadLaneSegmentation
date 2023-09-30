import logging
import os
from enum import Enum, auto

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .baseparser import BaseParser
from .masks import MaskProcessor
from .utils import fs
from .utils.hash import dir_hash
from .utils.torch_transform import image_2_tensor, mask_2_tensor


class DatasetMode(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class FileDataset:
    """Dataset class for file control functions."""

    default_extension = "xlsx"

    def __init__(self, dpath: str):
        self._logger = logging.getLogger(self.__class__.__name__)

        if not os.path.exists(dpath):
            raise ValueError("Dataset is not initialized - create it first!")

        self.dir_hash = dir_hash(dpath)
        self._dpath = dpath

        self._logger.info(f"Dataset directory: {self._dpath}")
        self._logger.info(f"Computed dataset hash: {self.dir_hash}")

        self._train_fpath = os.path.join(self._dpath, f"train.{self.default_extension}")
        self._logger.info(f"Train data path: {self._train_fpath}")

        self._valid_fpath = os.path.join(self._dpath, f"valid.{self.default_extension}")
        self._logger.info(f"Val data path: {self._valid_fpath}")

        self._test_fpath = os.path.join(self._dpath, f"test.{self.default_extension}")
        self._logger.info(f"Test data path: {self._test_fpath}")

    @property
    def hash(self) -> str:
        return self.dir_hash

    def _read_file(self, fpath: str) -> pd.DataFrame:
        if fpath.lower().endswith(".csv"):
            return pd.read_csv(fpath, index_col=0)
        elif fpath.lower().endswith(f".{self.default_extension}"):
            return pd.read_excel(fpath, index_col=0)

    def get_data(self, mode: DatasetMode) -> pd.DataFrame:
        if mode.value == DatasetMode.TRAIN.value:
            df = self._read_file(self._train_fpath)
            self._logger.info(f"Train Data Shape: {df.shape}")
            return df
        elif mode.value == DatasetMode.VALID.value:
            df = self._read_file(self._valid_fpath)
            self._logger.info(f"Valid Data Shape: {df.shape}")
            return df
        elif mode.value == DatasetMode.TEST.value:
            df = self._read_file(self._test_fpath)
            self._logger.info(f"Test Data Shape: {df.shape}")
            return df
        raise ValueError(f"Invalid mode value: {mode}")


class SegmentationDataset(Dataset):
    def __init__(
        self,
        dpath: str,
        df: pd.DataFrame,
        label_map: dict,
        transforms=None,
    ):
        super().__init__()

        self._dpath = dpath
        self._df = df
        self._label_map = label_map
        self._mask_processor = MaskProcessor()
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._df)

    def get_sample(self, img_fpath: str, mask_fpath: str) -> tuple[np.ndarray, np.ndarray]:
        img_fpath = os.path.join(self._dpath, img_fpath)
        mask_fpath = os.path.join(self._dpath, mask_fpath)

        img = fs.read_image(img_fpath)
        mask = fs.read_image(mask_fpath)

        if self._transforms is not None:
            img, mask = self._transforms(img, mask)

        label_mask = self._mask_processor.to_label_mask(mask, self._label_map)
        one_hot_mask = self._mask_processor.to_ohe_mask(label_mask, self._label_map)

        return img, one_hot_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._df.iloc[idx]

        img, ohe_masks = self.get_sample(row[BaseParser.src_key], row[BaseParser.target_key])

        # TODO: normalize img, values: [0, 1]

        img_tensor = image_2_tensor(img)
        mask_tensor = mask_2_tensor(ohe_masks)

        return {"features": img_tensor, "targets": mask_tensor}
