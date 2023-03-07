from typing import Optional

import logging

import numpy as np
import torch
from segmentation_models_pytorch import Unet
from tqdm import tqdm

from .masks import MaskProcessor
from .utils.image import glue_blocks, read_blocks
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
        self._mask_processor = MaskProcessor()

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

    def get_rgb_mask(self, label_map: dict) -> ImageRGB:
        label_mask = self.get_label_mask()
        rgb_mask = self._mask_processor.label_to_rgb(label_mask, label_map)
        return rgb_mask


class InferenceResult:
    def __init__(self, scores: list[SegmentationResult], label_map: dict):
        self._scores = scores
        self._label_map = label_map

    def get_label_patches(self):
        return [res.get_label_mask() for res in self._scores]

    def get_label_mask(self, img_shape: tuple[int, int]) -> ImageMask:
        label_patches = [res.get_label_mask() for res in self._scores]

        img_h, img_w = img_shape[:2]
        label_mask = glue_blocks(img_w, img_h, label_patches)
        return label_mask

    def get_rgb_mask(self, img_shape: tuple[int, int]) -> ImageRGB:
        rgb_cell_masks = [res.get_rgb_mask(self._label_map) for res in self._scores]

        img_h, img_w = img_shape[:2]
        rgb_mask = glue_blocks(img_w, img_h, rgb_cell_masks)
        return rgb_mask

    def get_heatmap(self, label_id: int, img_shape: tuple[int, int]) -> ImageMask:
        heatmaps = [np.clip(res.get_heatmap_mask(label_id), 0, 1) for res in self._scores]
        img_h, img_w = img_shape[:2]
        full_heatmap = glue_blocks(img_w, img_h, heatmaps, gray_scale=True)
        full_heatmap = (full_heatmap * 255).astype(np.uint8)
        return full_heatmap


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

    def predict(self, full_img: ImageRGB) -> ImageRGB:
        img_patches = read_blocks(full_img, self.cell_size)

        pred_seg_results = []

        stream = self.batch_generator(img_patches)
        for img_batch_t in stream:
            with torch.no_grad():
                img_batch_t = img_batch_t.to(self._device)

                logits_pred_t = self._model(img_batch_t)
                scores_pred_t = torch.softmax(logits_pred_t, dim=1)  # BCHW
                scores_pred_t = scores_pred_t.permute(0, 2, 3, 1)  # BHWC

                for batch_i in range(scores_pred_t.shape[0]):
                    pred_result = SegmentationResult(scores_pred_t[batch_i].cpu().numpy())
                    pred_seg_results.append(pred_result)

        pred_infer_result = InferenceResult(pred_seg_results, label_map=self.label_map)
        return pred_infer_result
