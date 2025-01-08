import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.registry import LOSS
from src.utils import powerset


LOSS["BCEWithLogitsLoss"] = nn.BCEWithLogitsLoss


# https://github.com/shuaizzZ/Dice-Loss-PyTorch/blob/master/dice_loss.py
@LOSS.register("Dice")    
class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6, batch=False):
        super().__init__()
        self.epsilon = epsilon
        self.batch = batch

    def forward(self, y_pred, y_true):
        # y_true shape: (B, 512, 512)
        # y_pred shape: (B, 512, 512)
        N = y_true.size(0) 
        y_pred = y_pred.view(N, -1)
        y_true = y_true.view(N, -1)        
        y_pred = F.sigmoid(y_pred)

        if self.batch:
            intersection = intersection = (y_pred * y_true).sum()
            # area_sum = y_pred.pow(2).sum() + y_true.sum()
            area_sum = y_pred.sum() + y_true.sum()
            dice_coefficient = (2 * intersection + self.epsilon) / (area_sum + self.epsilon)
            return 1 - dice_coefficient

        intersection = (y_pred * y_true).sum(dim=1)
        # area_sum = y_pred.pow(2).sum(dim=1) + y_true.sum(dim=1)
        area_sum = y_pred.sum(dim=1) + y_true.sum(dim=1)
        dice_coefficient = (2 * intersection + self.epsilon) / (area_sum + self.epsilon)
        return (1 - dice_coefficient).mean()
            


@LOSS.register("Mix")
class MixLoss(nn.Module):
    def __init__(self, weights=(0.3, 0.7), epsilon=1e-6, batch=True):
        super().__init__()
        self.weights = weights
        self._cross_entropy_loss = nn.BCEWithLogitsLoss()
        self._dice_loss = DiceLoss(epsilon=epsilon, batch=batch)
        
    def forward(self, y_pred, y_true):
        bce_loss = self._cross_entropy_loss(y_pred, y_true)
        dice_loss = self._dice_loss(y_pred, y_true)
        return self.weights[0] * bce_loss + self.weights[1] * dice_loss
    

@LOSS.register("tvMF_Mix")
class tvMF_MixLoss(nn.Module):
    def __init__(self, weights=(0.3, 0.7), kappa=2, batch=True):
        super().__init__()
        self.weights = weights
        self._cross_entropy_loss = nn.BCEWithLogitsLoss()
        self._dice_loss = tvMF_DiceLoss(batch=batch, kappa=kappa)
        
    def forward(self, y_pred, y_true):
        bce_loss = self._cross_entropy_loss(y_pred, y_true)
        dice_loss = self._dice_loss(y_pred, y_true)
        return self.weights[0] * bce_loss + self.weights[1] * dice_loss
    

@LOSS.register("MultiScale")
class MultiScaleLoss(nn.Module):
    def __init__(self, supervision="mutation", **kwargs):
        super().__init__()
        self.supervision = supervision
        kwargs["type"] = kwargs.pop("loss_type")
        self.loss_fn = LOSS.build(**kwargs)

        # self.ce_loss = nn.BCEWithLogitsLoss()
        # self.dice_loss = DiceLoss(epsilon=epsilon, batch=batch)

        self.make_idxs()        

    def forward(self, P, y_true):
        loss = 0.0
        for s in self.ss:
            iout = 0.0
            if(s==[]):
                continue
            for idx in range(len(s)):
                iout += P[s[idx]]

            # loss_ce = self.ce_loss(iout, y_true)
            # loss_dice = self.dice_loss(iout, y_true)
            # loss += (self.weights[0] * loss_ce + self.weights[1] * loss_dice)
            loss += self.loss_fn(iout, y_true)
        return loss

    def make_idxs(self):
        assert self.supervision in ("mutation", "deep_supervision")
        n_outs = 4
        out_idxs = list(np.arange(n_outs)) #[0, 1, 2, 3]#, 4, 5, 6, 7]
        if self.supervision == 'mutation':
            self.ss = [x for x in powerset(out_idxs)]

        elif self.supervision == 'deep_supervision':
            self.ss = [[x] for x in out_idxs]
        
        else:
            self.ss = [[-1]]



@LOSS.register("tvMF_Dice")
class tvMF_DiceLoss(nn.Module):
    def __init__(self, kappa=2, batch=False):
        super().__init__()
        self.kappa = kappa
        self.batch = batch

    def forward(self, y_pred, y_true):
        y_true = y_true.float()
        y_pred = F.sigmoid(y_pred)
        
        if self.batch:
            y_pred = F.normalize(y_pred, p=2, dim=(0, 1, 2))
            y_true = F.normalize(y_true, p=2, dim=(0, 1, 2))
            cosine = (y_pred * y_true).sum()
            intersect =  (1 + cosine).div(1. + (1 - cosine) * self.kappa) - 1.
            return (1 - intersect) ** 2.0
        
        N = y_true.size(0)
        y_pred = y_pred.view(N, -1)
        y_true = y_true.view(N, -1)
        y_pred = F.normalize(y_pred, p=2, dim=1)
        y_true = F.normalize(y_true, p=2, dim=1)
        cosine = y_pred * y_true
        intersect =  (1 + cosine).div(1. + (1 - cosine) * self.kappa) - 1.
        return ((1 - intersect) ** 2.0).mean()