from typing import Optional, Type

from accelerate import Accelerator
from functools import partial

from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

import pytorch_warmup as warmup

# helpers

def exists(v):
    return v is not None

# constants

ConstantLRScheduler = partial(LambdaLR, lr_lambda = lambda step: 1.)

# optimizer with scheduler + warmup

class OptimizerWithWarmupSchedule(nn.Module):
    def __init__(
        self,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: Optional[Type[_LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        warmup_steps: int = 0,
        max_grad_norm: Optional[float] = None
    ):
        super().__init__()
        self.max_grad_norm = max_grad_norm
        self.warmup = warmup.LinearWarmup(optimizer, warmup_period = warmup_steps)

        if exists(scheduler):
            self.scheduler = scheduler(optimizer, **scheduler_kwargs)
        else:
            self.scheduler = ConstantLRScheduler(optimizer)

        self.optimizer = optimizer

        self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)
        self.accelerator = accelerator

    def state_dict(self):
        return dict(
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            warmup = self.warmup.state_dict()
        )

    def load_state_dict(self, pkg):
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.scheduler.load_state_dict(pkg['scheduler'])
        self.warmup.load_state_dict(pkg['warmup'])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if exists(self.max_grad_norm):
            for param_group in self.optimizer.param_groups:
                self.accelerator.clip_grad_norm_(param_group['params'], self.max_grad_norm)

        self.optimizer.step()

        if not self.accelerator.optimizer_step_was_skipped:
            with self.warmup.dampening():
                self.scheduler.step()
