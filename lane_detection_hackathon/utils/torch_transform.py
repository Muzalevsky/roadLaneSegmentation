import numpy as np
import torch
import torch.nn.functional as F


def image_2_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image)
    tensor = tensor.float()
    tensor = tensor.div(255)
    tensor = tensor.permute(2, 0, 1)
    return tensor


def mask_2_tensor(mask: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(mask)
    tensor = tensor.float()
    tensor = tensor.permute(2, 0, 1)
    return tensor


def mask_batch_idx_2_ohe_encode(mask_batch: torch.LongTensor, n_classes: int) -> torch.LongTensor:
    mask_ohe = F.one_hot(mask_batch, n_classes)
    mask_ohe = mask_ohe.permute(0, 3, 1, 2)
    return mask_ohe
