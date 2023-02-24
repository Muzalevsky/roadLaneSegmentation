import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .baseparser import BaseParser
from .masks import MaskProcessor
from .utils import fs
from .utils.torch_transform import image_2_tensor, mask_2_tensor


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
        row = self._df.loc[idx]

        img, ohe_masks = self.get_sample(row[BaseParser.src_key], row[BaseParser.target_key])

        img_tensor = image_2_tensor(img)
        # TODO: check for multiple masks
        mask_tensor = mask_2_tensor(ohe_masks)

        return img_tensor, mask_tensor
