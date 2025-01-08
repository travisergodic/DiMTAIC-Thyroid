import logging

import torch


logger = logging.getLogger(__name__)


class AutocastAccumulateIterHook: 
    def __init__(self, max_norm=5, iter_to_accumulate=1):
        self.scaler = torch.cuda.amp.GradScaler()
        self.max_norm = max_norm
        self.iters_to_accumulate = iter_to_accumulate
        self.curr_iter = 0
    
    def run_iter(self, trainer, data, target):
        with torch.cuda.amp.autocast():
            pred = trainer.model(data)
            if isinstance(pred, list):
                pred = [t.squeeze(dim=1) for t in pred]
            else:
                pred = pred.squeeze(dim=1) 
                # B, 1, H, W --> B, H, W (Seg)
                # B, 1 --> B (Cls)
                
            loss = trainer.loss_fn(pred, target.float())
            loss = loss / self.iters_to_accumulate

        self.scaler.scale(loss).backward()

        # memory usage
        trainer.max_allocated = torch.cuda.max_memory_allocated() // (1024 ** 3)
        
        self.scaler.step(trainer.optimizer)
        self.scaler.update()
        trainer.optimizer.zero_grad()
        
        self.curr_iter += 1
        return loss.item()
        
    def reset(self, trainer):
        trainer.optimizer.zero_grad()
        self.curr_iter = 0


class SamIterHook: 
    def __init__(self, max_norm=None, iter_to_accumulate=None): 
        self.max_norm = max_norm
        self.iter_to_accumulate = iter_to_accumulate
        if max_norm is not None:
            logger.warning("max_norm is depreciated")
        if iter_to_accumulate is not None:
            logger.warning("iter_to_accumulate is depreciated")
    
    def run_iter(self, trainer, data, target):
        # first forward-backward
        pred = trainer.model(data)
        if isinstance(pred, list):
            pred = [t.squeeze(dim=1) for t in pred]
        else:
            pred = pred.squeeze(dim=1) 
            # B, 1, H, W --> B, H, W (Seg)
            # B, 1 --> B (Cls)
        loss = trainer.loss_fn(pred, target.float())
        loss.backward()
        trainer.optimizer.first_step(zero_grad=True)

        # second forward backward
        pred = trainer.model(data)
        if isinstance(pred, list):
            pred = [t.squeeze(dim=1) for t in pred]
        else:
            pred = pred.squeeze(dim=1) 
            # B, 1, H, W --> B, H, W (Seg)
            # B, 1 --> B (Cls)
        trainer.loss_fn(pred, target.float()).backward()

        trainer.max_allocated = torch.cuda.max_memory_allocated() // (1024 ** 3)

        trainer.optimizer.second_step(zero_grad=True)
        return loss.item()
        
    def reset(self, trainer):
        pass


def build_iter_hook(optimizer, **kwargs):
    if optimizer.__class__.__name__ == "SAM":
        return SamIterHook(**kwargs)
    return AutocastAccumulateIterHook(**kwargs)
