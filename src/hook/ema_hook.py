import os
import logging
from pathlib import Path

import torch
from timm.utils import ModelEmaV2

from src.registry import HOOK


# logger
logger = logging.getLogger(__name__)


@HOOK.register("ema_hook")
class EmaHook:
    def __init__(self, save_freq=1, decay=0.99):
        self.save_freq = save_freq
        self.decay = decay
        self.best_epoch = -1
        self.best_val = float("-inf")

    def before_train(self, trainer):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_ema = ModelEmaV2(trainer.model, self.decay, device=device)
        logger.info("Build ema model successfully")

    def after_iter(self, trainer):
        self.model_ema.update(trainer.model)

    def after_epoch(self, trainer):
        eval_dict = trainer.test(self.model_ema.module)
        logger.info(f"Epoch [{trainer.epoch}/{trainer.n_epochs}] ema_eval_dict: {eval_dict}")

        # save best
        if eval_dict[trainer.monitor] > self.best_val:
            self.best_val = eval_dict[trainer.monitor]
            self.best_epoch = trainer.epoch
            Path(trainer.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.model_ema.module.state_dict(), os.path.join(trainer.checkpoint_dir, "ema_best.pt"))
            logger.info(f"Save best ema_model at epoch {self.best_epoch}")
        
        # save last
        if trainer.epoch % self.save_freq == 0:
            Path(trainer.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.model_ema.module.state_dict(), os.path.join(trainer.checkpoint_dir, 'ema_last.pt'))