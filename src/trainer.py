import os
import logging
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

# logger
logger = logging.getLogger(__name__)

SAVE_BEST_START_EPOCH = 35

class Trainer:
    def __init__(
            self, model, optimizer, iter_hook, loss_fn, device, n_epochs, metric_list=None,
            save_freq=5, checkpoint_dir=None, monitor='Dice', hooks=None
        ):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.iter_hook = iter_hook
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.metric_list = metric_list
        self.best_epoch = 0
        self.epoch_train_records = defaultdict(list)
        self.epoch_test_records = defaultdict(list)
        self.hooks = hooks

    def fit(self, train_loader, test_loader=None):
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.call_hooks("before_train")

        for self.epoch in range(1, self.n_epochs + 1): 
            self.model.train()
            iter_train_loss_records = []
            
            self.call_hooks("before_epoch")

            for self.iter, batch in enumerate(self.train_loader):
                self.data, self.label = batch["data"].to(self.device), batch["label"].to(self.device)

                self.call_hooks("before_iter")
                train_loss = self.iter_hook.run_iter(self, self.data, self.label)
                self.call_hooks("after_iter")

                iter_train_loss_records.append(train_loss)

            self.iter_hook.reset(self)

            self.call_hooks("after_epoch")

            self.epoch_train_records['loss'].append(np.mean(iter_train_loss_records))
            curr_train_record = {k: v[-1] for k, v in self.epoch_train_records.items()}
            logger.info(f'Epoch [{self.epoch}/{self.n_epochs}] train_loss: {curr_train_record}')

            # evaluate
            if self.test_loader is not None:
                eval_dict = self.test(self.model)
                for met, val in eval_dict.items():
                    self.epoch_test_records[str(met)].append(val)

                logger.info(f"Epoch [{self.epoch}/{self.n_epochs}] eval_dict: {eval_dict}")
                
            # save best checkpoint
            if self.test_loader is not None:
                criterion = min if self.monitor == 'loss' else max
                if self.epoch >= SAVE_BEST_START_EPOCH:
                    if (self.epoch == SAVE_BEST_START_EPOCH) or (self.epoch_test_records[self.monitor][-1] == criterion(self.epoch_test_records[self.monitor][SAVE_BEST_START_EPOCH:])):
                        self.best_epoch = self.epoch
                        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'best.pt'))
                        logger.info(f"Save best model at epoch {self.best_epoch}")

            # save last checkpoint
            if self.epoch % self.save_freq == 0:
                Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'last.pt'))
        self.call_hooks("after_train")

    @torch.no_grad()
    def test(self, model):        
        model.eval()
        for batch in self.test_loader:
            data, label = batch["data"].to(device=self.device), batch["label"].to(device=self.device)
            # forward
            pred = model(data)
            # postprocess
            pred = torch.sigmoid(pred)

            for metric in self.metric_list: 
                metric.step(pred, label)
            
            # pred = (pred > 0.5).long()
        
        eval_dict = dict()
        for metric in self.metric_list:
            eval_dict[str(metric)] = metric.avg
            metric.reset()
        eval_dict["gpu_mem"] = self.max_allocated
        return eval_dict

    def call_hooks(self, name):
        if self.hooks is None:
            return 
        
        for hook in self.hooks:
            if hasattr(hook, name):
                getattr(hook, name)(self)