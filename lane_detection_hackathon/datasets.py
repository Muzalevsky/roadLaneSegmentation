import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import fs


class SegmentationDataset(Dataset):
    def __init__(
        self,
        dpath: str,
        df: pd.DataFrame,
        label_map: dict,
        transforms=None,
    ) -> None:
        super().__init__()

        self._dpath = dpath
        self._df = df
        self._label_map = label_map
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._df)

    def get_sample(self, img_fpath: str, mask_fpath: str) -> tuple[np.ndarray, np.ndarray]:
        print(img_fpath)
        print(mask_fpath)

        img_fpath = os.path.join(self._dpath, img_fpath)
        mask_fpath = os.path.join(self._dpath, mask_fpath)

        img = fs.read_image(img_fpath)
        mask = fs.read_mask(mask_fpath)

        if self._transforms is not None:
            img, mask = self._transforms(img, mask)

        # Convert mask to one-hot
        # one_hot_mask = np.zeros((*mask.shape[:2], len(self._label_map)))
        # for i, label_color in enumerate(self._label_map.values()):
        #     yy, xx, _zz = np.where(mask == label_color)

        #     print(yy.shape, xx.shape)

        #     one_hot_mask[yy, xx, i] = 1

        # Convert mask to one-hot
        one_hot_mask = np.zeros(
            (*mask.shape[:2], np.unique(mask).shape[0])
        )  # len(self._label_map)))
        for i, unique_value in enumerate(np.unique(mask)):
            one_hot_mask[:, :, i][mask == unique_value] = 1

        return img, mask, one_hot_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._df.loc[idx]

        img, src_mask, masks = self.get_sample(row["src"], row["tgt"])

        return img, src_mask, masks

        # img_tensor = image_2_tensor(img)
        # mask_tensor = mask_2_tensor(mask)

        # return img_tensor, mask_tensor
