import random
import logging

import torch.nn.functional as F

from src.registry import HOOK


# logger
logger = logging.getLogger(__name__)


@HOOK.register("seg_multiscale_hook")
class SegMutliscaleHook:
    def __init__(self, scales):
        self.scales = scales

    def before_iter(self, trainer):
        new_shape = random.choice(self.scales)
        trainer.data = F.interpolate(
            trainer.data, 
            size=new_shape, 
            mode='bilinear', 
            align_corners=False
        )
        trainer.label = F.interpolate(
            trainer.label.unsqueeze(1), 
            size=new_shape, mode='nearest'
        ).squeeze(1)


@HOOK.register("cls_multiscale_hook")
class ClsMutliscaleHook:
    def __init__(self, scales):
        self.scales = scales

    def before_iter(self, trainer):
        new_shape = random.choice(self.scales)
        trainer.data = F.interpolate(
            trainer.data, 
            size=new_shape, 
            mode='bilinear', 
            align_corners=False
        )