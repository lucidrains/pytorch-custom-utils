from functools import partial
from typing import Optional, Callable
from contextlib import nullcontext, contextmanager

from torch.nn import Module

from accelerate import Accelerator
from accelerate.tracking import WandBTracker

# helper functions

def exists(v):
    return v is not None

@contextmanager
def combine_contexts(a, b):
    with a() as c1, b() as c2:
        yield (c1, c2)

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

# automatically unwrap model when attribute cannot be found on the maybe ddp wrapped main model

class ForwardingWrapper:
  def __init__(self, parent, child):
    self.parent = parent
    self.child = child

  def __getattr__(self, key):
    if hasattr(self.parent, key):
      return getattr(self.parent, key)

    return getattr(self.child, key)

  def __call__(self, *args, **kwargs):
    call_fn = self.__getattr__('__call__')
    return call_fn(*args, **kwargs)

def auto_unwrap_model(
    accelerator_instance_name = 'accelerator',
    model_instance_name = 'model'
):
    def decorator(klass):
        _orig_init = klass.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            model = getattr(self, model_instance_name)
            accelerator = getattr(self, accelerator_instance_name)

            assert isinstance(accelerator, Accelerator)
            forward_wrapped_model = ForwardingWrapper(model, accelerator.unwrap_model(model))
            setattr(self, model_instance_name, forward_wrapped_model)

        klass.__init__ = __init__
        return klass

    return decorator

# gradient accumulation context manager
# for no_sync context on all but the last iteration

def model_forward_contexts(
    accelerator: Accelerator,
    model: Module,
    grad_accum_steps: int = 1
):
    for i in range(grad_accum_steps):
        is_last_step = i == grad_accum_steps - 1

        maybe_no_sync = partial(accelerator.no_sync, model) if not is_last_step else nullcontext

        yield partial(combine_contexts, accelerator.autocast, maybe_no_sync)
