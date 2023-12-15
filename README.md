## Pytorch Custom Utils (wip)

Just some miscellaneous utility functions / decorators / modules related to Pytorch and <a href="https://huggingface.co/docs/accelerate/index">Accelerate</a> to help speed up implementation of new AI research

## Install

```bash
$ pip install pytorch-custom-utils
```

### Quick save and load

Class decorator for adding a quick `save` and `load` method to the module instance. Can also initialize the entire network with a class method, `init_and_load`.

ex.

```python
import torch
from torch import nn

from pytorch_custom_utils import save_load

# decorate the entire class with `save_load` class decorator

@save_load()
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)

# instantiated mlp

mlp = MLP(dim = 512)

# now you have a save and load method

mlp.save('./mlp.pt')
mlp.load('./mlp.pt')

# you can also directly initialize from the checkpoint, without having to save the corresponding hyperparameters (in this case, dim = 512)

mlp = MLP.init_and_load('./mlp.pt')
```

### Keep track of device on module

ex.

```python
import torch
from torch import nn

from pytorch_custom_utils import module_device

# decorate the class with `module_device` class decorator

@module_device()
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, dim)

    def forward(self, x):
        return self.net(x)

# instantiated mlp

mlp = MLP(dim = 512)
mlp.to(torch.device('mps'))

# now you have a convenient .device

mlp.device # mps:0
```
