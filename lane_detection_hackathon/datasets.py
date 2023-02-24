import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .masks import MaskProcessor
from .utils import fs


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

        return img, mask, one_hot_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._df.loc[idx]

        img, masks = self.get_sample(row["src"], row["tgt"])

        return img, masks

        # img_tensor = image_2_tensor(img)
        # mask_tensor = mask_2_tensor(mask)

        # return img_tensor, mask_tensor
