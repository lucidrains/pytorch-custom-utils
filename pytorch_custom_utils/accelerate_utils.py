from typing import Optional, Callable

from contextlib import contextmanager

from accelerate import Accelerator
from accelerate.tracking import WandBTracker

def exists(v):
    return v is not None

def find_first(cond: Callable, arr):
    for el in arr:
        if cond(el):
            return el

    return None

# adds a context manager for wandb tracking with a specific project and experiment name

def add_wandb_tracker_contextmanager(
    accelerator_instance_name = 'accelerator',
    tracker_hps_instance_name = 'tracker_hps'
):
    def decorator(klass):

        @contextmanager
        def wandb_tracking(
            self,
            project: str,
            run: Optional[str] = None,
            hps: Optional[dict] = None
        ):
            maybe_accelerator = getattr(self, accelerator_instance_name, None)

            assert exists(maybe_accelerator) and isinstance(maybe_accelerator, Accelerator), f'Accelerator instance not found at self.{accelerator_instance_name}'

            hps = getattr(self, tracker_hps_instance_name, hps)

            maybe_accelerator.init_trackers(project, config = hps)

            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), maybe_accelerator.trackers)

            assert exists(wandb_tracker), 'wandb tracking was not enabled. you need to set `log_with = "wandb"` on your accelerate kwargs'

            if exists(run):
                assert exists(wandb_tracker)
                wandb_tracker.run.name = run

            yield

            maybe_accelerator.end_training() 

        if not hasattr(klass, 'wandb_tracking'):
            klass.wandb_tracking = wandb_tracking

        return klass

    return decorator
