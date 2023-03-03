from typing import Optional

import logging

import numpy as np
import torch
from segmentation_models_pytorch import Unet
from tqdm import tqdm

from .patch_extractor import ImageBlockReader
from .utils.torch_transform import image_2_tensor
from .utils.types import Dict, ImageMask, ImageRGB


class BaseInference:
    """Base Inference class implementation."""

    def __init__(self, model, batch_size, device):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._device = device
        self._model = model
        self._batch_size = batch_size

    def _prepare_model(self):
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(self._device)

        self._model.eval()
        self._model = self._model.to(self._device)

    @classmethod
    def from_file(cls, fpath: str, device=None, **kwargs):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_chk = torch.load(fpath, map_location=device)
        return cls.from_checkpoint(model_chk, device=device, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_state: dict, **kwargs):
        raise NotImplementedError()


class SegmentationResult:
    def __init__(self, pred_scores: np.ndarray):
        # NOTE: float16 for memory optimization
        self._scores_data = pred_scores.astype(np.float16)

    def get_label_mask(self) -> ImageMask:
        label_mask = np.argmax(self._scores_data, axis=-1).astype(np.uint8)
        return label_mask

    def get_heatmap_mask(self, label_id: int) -> ImageMask:
        probs = self._scores_data[:, :, label_id]

        label_mask = self.get_label_mask()
        bg_mask = label_mask == 0

        heatmap = np.zeros(probs.shape, dtype=np.float16)
        heatmap = np.where(bg_mask, -1, probs)

        return heatmap


class SegmentationInference(BaseInference):
    """Segmentation Inference class implementation."""

    def __init__(
        self,
        model,
        cfg: dict,
        batch_size: Optional[int] = 4,
        verbose: Optional[bool] = False,
        device: Optional[str] = "cpu",
    ):
        super().__init__(model, batch_size, device)

        self._block_maker = ImageBlockReader(pad_fill_value=(0, 0, 0))

        self._config = cfg
        self._verbose = verbose
        self._prepare_model()

    @property
    def cell_size(self):
        return self._config.get("cell_size")

    @property
    def label_map(self):
        return Dict(self._config.get("label_map"))

    @classmethod
    def from_checkpoint(cls, checkpoint_state: dict, **kwargs):
        model_state = checkpoint_state["model_state_dict"]
        model_cfg = checkpoint_state.get("config")

        n_classes = len(model_cfg["label_map"])
        # TODO: fix for another model, make more universal
        model = Unet(encoder_name="resnet34", classes=n_classes)

        model.load_state_dict(model_state)

        obj_ = cls(model=model, cfg=model_cfg, **kwargs)
        return obj_

    def batch_generator(self, img_patches: list[ImageRGB]):
        tensors_batch = []

        stream = img_patches
        if self._verbose:
            stream = tqdm(img_patches, desc="Batch Processing")

        for img in stream:
            img_t = image_2_tensor(img)
            tensors_batch.append(img_t)

            if len(tensors_batch) >= self._batch_size:
                batch_t = torch.stack(tensors_batch, axis=0)
                yield batch_t
                tensors_batch.clear()

        if len(tensors_batch):
            batch_t = torch.stack(tensors_batch, axis=0)
            yield batch_t
            tensors_batch.clear()

    def predict(self, full_img: ImageRGB) -> list[SegmentationResult]:
        img_patches = self._block_maker.read_blocks(full_img, self.cell_size)

        pred_results = []

        stream = self.batch_generator(img_patches)
        for img_batch_t in stream:
            with torch.no_grad():
                img_batch_t = img_batch_t.to(self._device)

                logits_pred_t = self._model(img_batch_t)
                scores_pred_t = torch.softmax(logits_pred_t, dim=1)  # BCHW
                scores_pred_t = scores_pred_t.permute(0, 2, 3, 1)  # BHWC

                for batch_i in range(scores_pred_t.shape[0]):
                    pred_result = SegmentationResult(scores_pred_t[batch_i].cpu().numpy())
                    pred_results.append(pred_result)

        # TODO: merge cells into one image (?)
        return pred_results
