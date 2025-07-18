"""
Trainer
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from ..models.focal_loss import FocalLoss
from ..config import LR, PATIENCE, MAX_EPOCHS

class MLPTrainer:
    def __init__(self, model, train_ds, val_ds, lr=LR, batch_size_train=256, batch_size_val=128):
        self.model = model
        self.train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=batch_size_val)
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'max', patience=5, factor=0.5)
        self.crit = FocalLoss()

    # -------- Single epoch training --------
    def train_epoch(self, device):
        self.model.train()
        loss_sum = 0
        for xb, yb in self.train_loader:
            xb, yb = xb.to(device), yb.to(device)
            self.opt.zero_grad()
            loss = self.crit(self.model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            loss_sum += loss.item() * xb.size(0)
        return loss_sum / len(self.train_loader.dataset)

    # -------- Evaluate --------
    @torch.no_grad()
    def evaluate(self, loader, device):
        self.model.eval()
        correct, outputs = 0, []
        for xb, yb in loader:
            logits = self.model(xb.to(device))
            outputs.append(logits.cpu())
            correct += (logits.argmax(1).cpu() == yb).sum().item()
        return correct / len(loader.dataset), torch.cat(outputs)
