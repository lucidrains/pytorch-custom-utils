from torch.nn import Module

# provides a .total_parameters property for your model
# that simply sums all the parameters across modules

def total_parameters(
    count_only_requires_grad = False,
    total_parameters_property_name = 'total_parameters'
):
    def decorator(klass):
        assert issubclass(klass, Module), 'should decorate a subclass of torch.nn.Module'

        @property
        def _total_parameters(self):
            return sum(p.numel() for p in self.parameters())

        @property
        def _total_parameters_with_requires_grad(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        fn = _total_parameters_with_requires_grad if count_only_requires_grad else  _total_parameters

        setattr(klass, total_parameters_property_name, fn)
        return klass

    return decorator
