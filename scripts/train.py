import logging
import os
from functools import partial

import hydra
import torch
from catalyst import dl
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from road_lane_segmentation import callbacks as cb
from road_lane_segmentation.datasets import DatasetMode, FileDataset, SegmentationDataset
from road_lane_segmentation.utils import misc, project

CURRENT_DPATH = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DPATH, os.pardir))
CONFIG_DPATH = os.path.join(PROJECT_ROOT, "configs")
# TODO: fix hardcode
DATA_DPATH = os.path.join(PROJECT_ROOT, "data")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lane_marks_segmentation")


def get_loaders(cfg: DictConfig, file_dataset: FileDataset):
    num_workers = project.get_n_workers(cfg.num_workers, logger)
    if num_workers > cfg.batch_size:
        logger.info(f"Redefine `num_workers` from {num_workers} to {cfg.batch_size}")
        num_workers = cfg.batch_size

    try:
        transformer = hydra.utils.instantiate(cfg.transformer)
    except Exception:
        transformer = None

    train_df = file_dataset.get_data(mode=DatasetMode.TRAIN)
    valid_df = file_dataset.get_data(mode=DatasetMode.VALID)

    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Valid data shape: {valid_df.shape}")

    files_dpath = os.path.join(DATA_DPATH, cfg.dataset_name)

    train_dataset = SegmentationDataset(
        dpath=files_dpath,
        df=train_df,
        label_map=cfg.label_map,
        transforms=transformer,
    )
    valid_dataset = SegmentationDataset(
        dpath=files_dpath,
        df=valid_df,
        label_map=cfg.label_map,
        transforms=None,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Valid dataset size: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=True,
        # To avoid batch_norm fails
        # drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return {"train": train_loader, "valid": valid_loader}


@hydra.main(config_path=CONFIG_DPATH, config_name="train")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("len", lambda data: len(data))

    dataset_dpath = os.path.join(DATA_DPATH, cfg.dataset_name, cfg.dataset_version)
    logger.info(f"Dataset dpath: {dataset_dpath}")
    file_dataset = FileDataset(dataset_dpath)

    clearml_task = misc.init_clearml(cfg, file_dataset.hash)

    checkpoint_save_dict = dict(
        config=OmegaConf.to_container(cfg, resolve=True),
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
    )
    checkpoints_suffix = f"_{clearml_task.id}" if clearml_task is not None else ""
    logger.info(f"Custom checkpoints suffix: {checkpoints_suffix}")

    checkpoint_dpath = project.setup_train_project(PROJECT_ROOT, seed=cfg.seed)

    LOG_ON_BATCH = False
    LOGS_DPATH = os.path.join(os.getcwd(), "logs")

    label_names = list(cfg.label_map.keys())
    num_classes = len(label_names)
    logger.info(f"Number of classes: {num_classes}")

    if clearml_task is not None:
        clearml_task.add_tags([f"{num_classes} classes"])

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    weights = list(cfg.class_weights.values())
    weights = torch.tensor(weights)
    criterion = hydra.utils.instantiate(cfg.criterion, weight=weights)

    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )

    additional_callbacks = [
        cb.LogBestCheckpoint2ClearMLCallback(
            logdir=checkpoint_dpath,
            loader_key="valid",
            metric_key="dice",
            minimize=False,
            save_kwargs=checkpoint_save_dict,
            clearml_task=clearml_task,
            suffix=checkpoints_suffix,
        ),
        cb.LogBestCheckpoint2ClearMLCallback(
            logdir=checkpoint_dpath,
            loader_key="valid",
            metric_key="iou",
            minimize=False,
            save_kwargs=checkpoint_save_dict,
            clearml_task=clearml_task,
            suffix=checkpoints_suffix,
        ),
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=get_loaders(cfg, file_dataset),
        num_epochs=cfg.num_epochs,
        callbacks=[
            dl.BatchTransformCallback(
                input_key="logits",
                output_key="scores",
                scope="on_batch_end",
                transform=partial(nn.functional.softmax, dim=1),
            ),
            # This one requires output in format [0; 1]
            dl.IOUCallback(
                input_key="scores",
                target_key="targets",
                class_names=label_names,
                log_on_batch=LOG_ON_BATCH,
            ),
            # This one requires output in format [0; 1]
            dl.DiceCallback(
                input_key="scores",
                target_key="targets",
                class_names=label_names,
                log_on_batch=LOG_ON_BATCH,
            ),
            # This one requires output in format [0; 1]
            dl.TrevskyCallback(
                input_key="scores",
                target_key="targets",
                class_names=label_names,
                alpha=0.2,
                log_on_batch=LOG_ON_BATCH,
            ),
            dl.OptimizerCallback(
                metric_key="loss",
                accumulation_steps=1,
                grad_clip_fn=nn.utils.clip_grad_norm_,
                grad_clip_params=dict(max_norm=1, norm_type=2),
            ),
            dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            *additional_callbacks,
        ],
        logdir=LOGS_DPATH,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=cfg.verbose,
        load_best_on_end=True,
        timeit=cfg.time_profiling,
    )


if __name__ == "__main__":
    main()
