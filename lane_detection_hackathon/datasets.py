import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import fs
from .utils.torch_transform import image_2_tensor, mask_2_tensor


class SegmentationDataset(Dataset):
    def __init__(
        self,
        fpaths_list: list[tuple[str, str]],
        n_classes: int,
        transforms=None,
    ) -> None:
        super().__init__()

        self._fpaths = fpaths_list
        self._n_classes = n_classes
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._fpaths)

    def get_sample(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img_fpath, mask_fpath = self._fpaths[idx]

        img = fs.read_image(img_fpath)
        mask = fs.read_mask(mask_fpath)

        if self._transforms is not None:
            img, mask = self._transforms(img, mask)

        # Convert mask to one-hot
        one_hot_mask = np.zeros((*mask.shape[:2], self._n_classes))
        for i, unique_value in enumerate(np.unique(mask)):
            one_hot_mask[:, :, i][mask == unique_value] = 1

        return img, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.get_sample(idx)

        img_tensor = image_2_tensor(img)
        mask_tensor = mask_2_tensor(mask)

        return img_tensor, mask_tensor
