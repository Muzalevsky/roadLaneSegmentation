import logging

import torch


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


class SegmentationInference(BaseInference):
    # TODO: inference implementation with batch-generation
    pass
