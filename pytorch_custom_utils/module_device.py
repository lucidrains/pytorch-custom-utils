from functools import wraps
from typing import List
from optree import tree_flatten, tree_unflatten

import torch
from torch import is_tensor
from torch.nn import Module

# provides a .device for your model
# uses a dummy scalar tensor

def module_device(
    device_property_name = 'device'
):
    def decorator(klass):
        assert issubclass(klass, Module), 'should decorate a subclass of torch.nn.Module'

        _orig_init = klass.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)

            self.register_buffer('_dummy', torch.tensor(0), persistent = False)

        @property
        def _device_property(self):
            return self._dummy.device

        klass.__init__ = __init__
        setattr(klass, device_property_name, _device_property)
        return klass

    return decorator

# a decorator that automatically casts all tensors coming into .forward to the proper device

def autocast_device(
    methods: List[str] = ['forward']
):
    def decorator(klass):
        assert issubclass(klass, Module), 'should decorate a subclass of torch.nn.Module'

        orig_fns = [getattr(klass, method) for method in methods]

        for method, orig_fn in zip(methods, orig_fns):

            @wraps(orig_fn)
            def fn(self, *args, **kwargs):

                # determine device
                # use dummy from decorator above
                # otherwise look for parameters and use the device on that

                if hasattr(self, '_dummy'):
                    device = self._dummy.device
                else:
                    device = next(self.parameters()).device

                # flatten

                flattened_args, tree_spec = tree_flatten([args, kwargs])

                # transform args

                maybe_transformed_args = []

                for flattened_arg in flattened_args:
                    if is_tensor(flattened_arg):
                        flattened_arg = flattened_arg.to(device)

                    maybe_transformed_args.append(flattened_arg)

                # unflatten

                args, kwargs = tree_unflatten(tree_spec, maybe_transformed_args)

                # call original fn

                orig_fn(self, *args, **kwargs)

            setattr(klass, method, fn)

        return klass

    return decorator
