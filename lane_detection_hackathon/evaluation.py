from typing import Optional

import logging
import os

import pandas as pd
from inference import SegmentationInference
from masks import MaskProcessor
from metrics import MetricCalculator
from tqdm import tqdm
from utils import fs
from utils.types import ImageMask


class Evaluator:
    """Implementation of inference evaluation."""

    fname_key = "fname"

    def __init__(
        self, dpath: str, infer_model: SegmentationInference, verbose: Optional[bool] = False
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._mask_processor = MaskProcessor()

        self._dpath = dpath
        self._infer_model = infer_model

        self._eval_data = []

        self._verbose = verbose

    @property
    def eval_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._eval_data)

    def _process_masks(
        self, metric_calculator: MetricCalculator, true_mask: ImageMask, pred_mask: ImageMask
    ):
        # TODO: split masks into patches
        true_patches = None
        pred_patches = None

        # TODO: generator?
        for true_patch, pred_patch in zip(true_patches, pred_patches):
            true_patch_label = self._mask_processor.to_label_mask(true_patch)
            true_patch_ohe = self._mask_processor.to_ohe_mask(true_patch_label)

            # TODO: check pred label mask
            pred_patch_ohe = self._mask_processor.to_ohe_mask(pred_patch)

            metric_calculator.update(true_patch_ohe, pred_patch_ohe)
            # TODO: save masks (?)

    def evaluate(self, img_fpaths: list[str], true_mask_fpaths: list[str]):
        stream = zip(img_fpaths, true_mask_fpaths)
        if self._verbose:
            # TODO: check, add total
            stream = tqdm(stream, desc="Evaluation")

        class_names = self._infer_model.label_map.keys()
        for img_fpath, true_mask_fpath in stream:
            metric_calculator = MetricCalculator(class_names)

            img_fname = os.path.basename(img_fpath)
            abs_img_fpath = os.path.join(self._dpath, img_fpath)
            abs_true_mask_fpath = os.path.join(self._dpath, true_mask_fpath)

            img = fs.read_image(abs_img_fpath)
            true_mask = fs.read_image(abs_true_mask_fpath)

            pred_mask = self._infer_model.predict(img)

            self._process_masks(metric_calculator, true_mask, pred_mask)

            iou_file_metrics = metric_calculator.compute_iou()
            dice_file_metrics = metric_calculator.compute_dice()

            file_metrics = iou_file_metrics | dice_file_metrics
            file_metrics[self.fname_key] = img_fname

            self._eval_data.append(file_metrics)

    def get_average_metrics(self) -> pd.DataFrame:
        return self.eval_df.mean(axis=0)
