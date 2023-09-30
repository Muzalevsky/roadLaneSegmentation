from clearml import Task
from omegaconf import DictConfig, OmegaConf

from .hash import dict_hash


def init_clearml(cfg: DictConfig, dataset_hash: str):
    if not cfg.enable_clearml:
        return None

    cfg_hash = dict_hash(OmegaConf.to_container(cfg, resolve=True))

    tags = []

    if "logger_tags" in cfg:
        for tag in cfg.logger_tags:
            tags.append(tag)

    if "model" in cfg:
        model_name = cfg.model._target_.split(".")[-1]
        tags.append(model_name)

        if "n_stages" in cfg.model:
            tags.append(f"stages {cfg.model.n_stages}")

    experiment_name = f"{cfg.dataset_version}-{dataset_hash[:8]}-{cfg_hash[:8]}"

    clearml_task = Task.init(
        project_name=cfg.project_name,
        task_name=experiment_name,
        reuse_last_task_id=False,
    )
    clearml_task.add_tags(tags)

    description = f"Dataset hash: {dataset_hash}\n"
    description += f"Config hash: {cfg_hash}\n"

    clearml_task.set_comment(description)

    return clearml_task
