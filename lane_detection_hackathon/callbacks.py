from pathlib import Path

from catalyst.callbacks.checkpoint import ICheckpointCallback, _save_checkpoint
from catalyst.core.callback import CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.extras.metric_handler import MetricHandler
from clearml import Task


class LogBestCheckpoint2ClearMLCallback(ICheckpointCallback):
    def __init__(
        self,
        logdir: str,
        # model selection info
        loader_key: str,
        metric_key: str,
        clearml_task: Task = None,
        minimize: bool = None,
        min_delta: float = 1e-6,
        # Additional data to save to checkpoint
        save_kwargs: dict = None,
        suffix: str = "",
    ):
        """Init."""
        super().__init__(order=CallbackOrder.external, node=CallbackNode.all)

        if loader_key is not None or metric_key is not None:
            assert loader_key is not None and metric_key is not None, (
                "For checkpoint selection `CheckpointCallback` "
                "requires both `loader_key` and `metric_key` specified."
            )
            self._use_model_selection = True
            self.minimize = minimize if minimize is not None else True  # loss-oriented selection
        else:
            self._use_model_selection = False
            self.minimize = False  # epoch-num-oriented selection

        self.logdir = logdir
        self.loader_key = loader_key
        self.metric_key = metric_key
        self.is_better = MetricHandler(minimize=minimize, min_delta=min_delta)
        self.best_score = None

        self._save_kwargs = save_kwargs if save_kwargs is not None else dict()
        self._clearml_task = clearml_task
        self._suffix = suffix

    def _pack_checkpoint(self, runner: "IRunner"):
        additional_info = dict(epoch=runner.stage_epoch_step)
        checkpoint = runner.engine.pack_checkpoint(
            model=runner.model, _info=additional_info, **self._save_kwargs
        )
        return checkpoint

    def _save_checkpoint(
        self, runner: IRunner, checkpoint: dict, is_best: bool, is_last: bool
    ) -> str:
        logdir = Path(f"{self.logdir}/")
        metric_name = self.metric_key.replace("/", "_")
        checkpoint_path = _save_checkpoint(
            runner=runner,
            logdir=logdir,
            checkpoint=checkpoint,
            suffix=f"best-{self.loader_key}-{metric_name}{self._suffix}",
        )

        return checkpoint_path

    def on_epoch_end(self, runner: "IRunner") -> None:
        """
        Collects and saves checkpoint after epoch.

        Args:
            runner: current runner
        """
        if runner.is_infer_stage:
            return
        if runner.engine.is_ddp and not runner.engine.is_master_process:
            return

        loader_metrics = runner.epoch_metrics[self.loader_key]
        if self.metric_key not in loader_metrics:
            return

        if self._use_model_selection:
            # score model based on the specified metric
            score = runner.epoch_metrics[self.loader_key][self.metric_key]
        else:
            # score model based on epoch number
            score = runner.global_epoch_step

        is_best = False
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            is_best = True

        if not is_best:
            # Save only best!
            return

        # pack checkpoint
        checkpoint = self._pack_checkpoint(runner)
        # save checkpoint
        checkpoint_path = self._save_checkpoint(
            runner=runner, checkpoint=checkpoint, is_best=is_best, is_last=True
        )

        if self._clearml_task is not None:
            metric_name = self.metric_key.replace("/", "_")
            self._clearml_task.upload_artifact(
                name=f"Best {self.loader_key}-{metric_name} model",
                artifact_object=checkpoint_path,
            )
