import logging

import torch
import timm.scheduler

from src.registry import HOOK


logger = logging.getLogger(__name__)


@HOOK.register("epoch_scheduler_hook")
class EpochSchedulerHook:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def before_train(self, trainer):
        # build scheduler
        kwargs = self.kwargs.copy()
        self.scheduler = eval(kwargs.pop("scheduler_type"))(trainer.optimizer, **kwargs)
        self.backend = self.kwargs["scheduler_type"].split(".")[0]

    def after_epoch(self, trainer):
        curr_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.epoch_train_records["lr"].append(curr_lr)

        if self.backend == "torch":
            self.scheduler.step()
        
        elif self.backend == "timm": 
            self.scheduler.step(trainer.epoch)

        else:
            logger.warning(f"unknown scheduler backend: {self.backend}")