
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    α-balanced, γ-focused Focal Loss
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        fl = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if   self.reduction == 'mean': return fl.mean()
        elif self.reduction == 'sum' : return fl.sum()
        else                         : return fl
