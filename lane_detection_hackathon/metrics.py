from typing import Callable

import logging
from collections import defaultdict
from functools import partial

import numpy as np

from .utils.types import ImageMask


def iou(tp: np.array, fp: np.array, fn: np.array, eps: float = 1e-7) -> np.array:
    union = tp + fp + fn
    score = (tp + eps * (union == 0).astype(np.float64)) / (tp + fp + fn + eps)
    return score


def dice(tp: np.array, fp: np.array, fn: np.array, eps: float = 1e-7) -> np.array:
    union = tp + fp + fn
    score = (2 * tp + eps * (union == 0).astype(np.float64)) / (2 * tp + fp + fn + eps)
    return score


class MetricCalculator:
    """Implementation of segmentation metric calculation."""

    def __init__(self, class_names: list[str]):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._class_names = class_names
        self._data = defaultdict(dict)

    @property
    def data(self) -> dict:
        return dict(self._data)

    def _compare_masks(self, true_mask: ImageMask, pred_mask: ImageMask) -> tuple[int, int, int]:
        if true_mask.shape != pred_mask.shape:
            raise ValueError(
                f"target mask shape ({true_mask.shape}) and "
                f"pred mask shape ({pred_mask.shape}) must be the same"
            )

        n_dims = len(pred_mask.shape)
        dims = list(range(n_dims))

        sum_per_class = partial(np.sum, axis=tuple(dims))
        tp = sum_per_class(pred_mask * true_mask)
        class_union = sum_per_class(pred_mask) + sum_per_class(true_mask)
        class_union -= tp
        fp = sum_per_class(pred_mask * (1 - true_mask))
        fn = sum_per_class(true_mask * (1 - pred_mask))
        return tp, fp, fn

    def update(self, true_mask: ImageMask, pred_mask: ImageMask):
        tp, fp, fn = self._compare_masks(true_mask, pred_mask)
        for idx, (tp_class, fp_class, fn_class) in enumerate(zip(tp, fp, fn)):
            self._data[idx]["tp"] += tp_class
            self._data[idx]["fp"] += fp_class
            self._data[idx]["fn"] += fn_class

    def _compute_metrics(self, metric_name: str, metric_func: Callable):
        macro_metric = 0
        per_class = []
        total_statistics = {}

        for class_data in self._data.values():
            metric_val = metric_func(**class_data)
            per_class.append(metric_val)
            macro_metric += metric_val

            for class_name, val in class_data.items():
                total_statistics[class_name] = total_statistics.get(class_name, 0) + val

        macro_metric /= len(self._data)
        micro_metric = metric_func(**total_statistics)

        metrics = {}
        for class_idx, val in enumerate(per_class):
            metrics[f"{metric_name}/{self._class_names[class_idx]}"] = val

        metrics[f"{metric_name}/micro"] = micro_metric
        metrics[f"{metric_name}/macro"] = macro_metric

        return metrics

    def compute_iou(self):
        return self._compute_metrics("iou", iou)

    def compute_dice(self):
        return self._compute_metrics("dice", dice)
