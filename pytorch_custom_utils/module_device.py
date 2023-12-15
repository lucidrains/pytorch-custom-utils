import torch
from torch.nn import Module

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
