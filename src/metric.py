import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F

from src.registry import METRIC
from src.utils import AverageMeter
    

@METRIC.register("IOU")
class IOU(AverageMeter):
    def __init__(self, epsilon=1): 
        super().__init__()
        self.epsilon = epsilon
        
    def __call__(self, y_hat, y_true):
        """
        C, H, W
        """
        y_true = torch.where(y_true==2, 1, y_true)
        y_hat = torch.where(y_hat==2, 1, y_hat)

        intersection = (y_true == y_hat) & (y_true == 1)
        union = (y_true == 1) | (y_hat == 1)  
        return (intersection / (union + self.epsilon)).mean().item()
    
    def __repr__(self) -> str:
        return "IOU"
    

@METRIC.register("Dice")
class DiceScore(AverageMeter):
    def __init__(self, threshold=0.5, epsilon=1e-6):
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon

    def __call__(self, y_hat, y_true):
        N = y_true.size(0)
        y_true = y_true.view(N, -1)
        y_hat = (y_hat.view(N, -1) > self.threshold).long()
        intersection_area = ((y_true == y_hat) & (y_true == 1)).sum(dim=1)
        area_sum = (y_true == 1).sum(dim=1) + (y_hat == 1).sum(dim=1)
        dice_coefficient = 2 * (intersection_area + self.epsilon) / (area_sum + self.epsilon)
        return dice_coefficient.mean().item()

    def __repr__(self) -> str:
        return "Dice"
    

@METRIC.register("Accuracy")
class Accuracy(AverageMeter):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    
    def __call__(self, y_hat, y_true):
        y_true = y_true.view(-1)
        y_hat = (y_hat.view(-1) > self.threshold).long()
        return (y_true == y_hat).float().mean().item()
    
    def __repr__(self) -> str:
        return "Accuracy"
    

@METRIC.register("F1")
class F1:
    def __init__(self, epsilon=1e-6, threshold=0.5) -> None:
        self.epsilon = epsilon
        self.threshold = threshold
        self.prob_list = []
        self.labels = []
        
    def step(self, y_hat, y_true):
        self.prob_list += y_hat.view(-1).tolist()
        self.labels += y_true.view(-1).tolist()

    @property
    def avg(self):
        y_pred = (np.array(self.prob_list) > self.threshold).astype(int)
        y_true = np.array(self.labels)
        return f1_score(y_true, y_pred, average='binary')
    
    def reset(self):
        self.prob_list = []
        self.labels = []

    def __repr__(self) -> str:
        return "F1"        
    


@METRIC.register("AUC")
class AUC:
    def __init__(self) -> None:
        self.prob_list = []
        self.labels = []
        
    def step(self, y_hat, y_true):
        self.prob_list += y_hat.view(-1).tolist()
        self.labels += y_true.view(-1).tolist()

    @property
    def avg(self):
        y_pred = np.array(self.prob_list)
        y_true = np.array(self.labels)
        return roc_auc_score(y_true, y_pred)
    
    def reset(self):
        self.prob_list = []
        self.labels = []

    def __repr__(self) -> str:
        return "AUC"