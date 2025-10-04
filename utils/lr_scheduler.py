import torch
from torch.optim import Optimizer
from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler

class LearningRateScheduler(_LRScheduler):
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr
    def step(self, *args, **kwargs):
        raise NotImplementedError
    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr
    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

class ReduceLROnPlateauScheduler(LearningRateScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            lr: float,
            patience: int = 1,
            factor: float = 0.3,
    ) -> None:
        super(ReduceLROnPlateauScheduler, self).__init__(optimizer, lr)
        self.lr = lr
        self.patience = patience
        self.factor = factor
        self.val_loss = 100.0
        self.count = 0

    def step(self, val_loss: float):
        if val_loss is not None:
            if self.val_loss <= val_loss:
                self.count += 1
            else:
                self.count = 0
                self.val_loss = val_loss
            if self.patience == self.count:
                self.count = 0
                self.lr *= self.factor
                self.set_lr(self.optimizer, self.lr)
        return self.lr

class WarmupLRScheduler(LearningRateScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            warmup_steps: int,
    ) -> None:
        super(WarmupLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr = init_lr
        if warmup_steps != 0:
            warmup_rate = peak_lr - init_lr
            self.warmup_rate = warmup_rate / warmup_steps
        else:
            self.warmup_rate = 0
        self.update_steps = 1
        self.lr = init_lr
        self.warmup_steps = warmup_steps

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        if self.update_steps <= self.warmup_steps:
            lr = self.init_lr + self.warmup_rate * self.update_steps
            self.set_lr(self.optimizer, lr)
            self.lr = lr
        self.update_steps += 1
        return self.lr

class WarmupReduceLROnPlateauScheduler(LearningRateScheduler):
    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            warmup_steps: int,
            patience: int = 1,
            factor: float = 0.3,
    ) -> None:
        super(WarmupReduceLROnPlateauScheduler, self).__init__(optimizer, init_lr)
        self.warmup_steps = warmup_steps
        self.update_steps = 0
        self.warmup_rate = (peak_lr - init_lr) / self.warmup_steps \
            if self.warmup_steps != 0 else 0
        self.schedulers = [
            WarmupLRScheduler(
                optimizer=optimizer,
                init_lr=init_lr,
                peak_lr=peak_lr,
                warmup_steps=warmup_steps,
            ),
            ReduceLROnPlateauScheduler(
                optimizer=optimizer,
                lr=peak_lr,
                patience=patience,
                factor=factor,
            ),
        ]

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps: return 0, self.update_steps
        else: return 1, None

    def step(self, val_loss: Optional[float] = None):
        stage, _ = self._decide_stage()
        if stage == 0: self.schedulers[0].step()
        elif stage == 1: self.schedulers[1].step(val_loss)
        self.update_steps += 1
        return self.get_lr()