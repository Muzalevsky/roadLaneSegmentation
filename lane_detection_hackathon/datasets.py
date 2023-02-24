import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import fs
from .utils.torch_transform import image_2_tensor, mask_2_tensor


class SegmentationDataset(Dataset):
    def __init__(
        self,
        dpath: str,
        fpaths_list: list[tuple[str, str]],
        label_map: dict,
        transforms=None,
    ) -> None:
        super().__init__()

        self._dpath = dpath
        self._fpaths = fpaths_list
        self._label_map = label_map
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._fpaths)

    def get_sample(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img_fpath, mask_fpath = self._fpaths[idx]

        img_fpath = os.path.join(self._dpath, img_fpath)
        mask_fpath = os.path.join(self._dpath, mask_fpath)

        img = fs.read_image(img_fpath)
        mask = fs.read_image(mask_fpath)

        if self._transforms is not None:
            img, mask = self._transforms(img, mask)

        # Convert mask to one-hot
        one_hot_mask = np.zeros((*mask.shape[:2], len(self._label_map)))
        for i, label_color in enumerate(self._label_map.values()):
            yy, xx, _zz = np.where(mask == label_color)
            one_hot_mask[yy, xx, i] = 1

        return img, one_hot_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.get_sample(idx)

        return img, mask

        img_tensor = image_2_tensor(img)
        mask_tensor = mask_2_tensor(mask)

        return img_tensor, mask_tensor
