import numpy as np
import torch


def image_2_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image)
    tensor = tensor.float()
    tensor = tensor.div(255)
    tensor = tensor.permute(2, 0, 1)
    return tensor


def mask_2_tensor(mask: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(mask)
    tensor = tensor.long()
    tensor = tensor.permute(2, 0, 1)
    return tensor
