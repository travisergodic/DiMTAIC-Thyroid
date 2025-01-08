import random
import logging

import torch

from src.registry import HOOK


# logger
logger = logging.getLogger(__name__)


@HOOK.register("mixup_hook")
class MixupHook:
    def __init__(self, prob=1.0, alpha=0.1):
        self.prob = prob
        self.alpha = alpha

    def before_iter(self, trainer):
        if random.uniform(0, 1) > self.prob:
            return

        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1

        data, label = trainer.data, trainer.label
        perm_indices = torch.randperm(data.size(0))
        trainer.data = lam * data + (1 - lam) * data[perm_indices, :]
        trainer.label = lam * label + (1 - lam) * label[perm_indices]