import pickle
from pathlib import Path
from packaging import version

import torch
from torch.nn import Module

from beartype import beartype
from beartype.typing import Optional

# helpers

def exists(v):
    return v is not None

@beartype
def save_load(
    save_method_name = 'save',
    load_method_name = 'load',
    config_instance_var_name = '_config',
    init_and_load_classmethod_name = 'init_and_load',
    version: Optional[str] = None
):
    def _save_load(klass):
        assert issubclass(klass, Module), 'save_load should decorate a subclass of torch.nn.Module'

        _orig_init = klass.__init__

        def __init__(self, *args, **kwargs):
            _config = pickle.dumps((args, kwargs))

            setattr(self, config_instance_var_name, _config)
            _orig_init(self, *args, **kwargs)

        def _save(self, path, overwrite = True):
            path = Path(path)
            assert overwrite or not path.exists()

            pkg = dict(
                model = self.state_dict(),
                config = getattr(self, config_instance_var_name),
                version = version,
            )

            torch.save(pkg, str(path))

        def _load(self, path, strict = True):
            path = Path(path)
            assert path.exists()

            pkg = torch.load(str(path), map_location = 'cpu')

            if exists(version) and exists(pkg['version']) and version.parse(version) != version.parse(pkg['version']):
                self.print(f'loading saved model at version {pkg["version"]}, but current package version is {__version__}')

            self.load_state_dict(pkg['model'], strict = strict)

        # init and load from
        # looks for a `config` key in the stored checkpoint, instantiating the model as well as loading the state dict

        @classmethod
        def _init_and_load_from(cls, path, strict = True):
            path = Path(path)
            assert path.exists()
            pkg = torch.load(str(path), map_location = 'cpu')

            assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

            config = pickle.loads(pkg['config'])
            args, kwargs = config
            model = cls(*args, **kwargs)

            _load(model, path, strict = strict)
            return model

        # set decorated init as well as save, load, and init_and_load

        klass.__init__ = __init__
        setattr(klass, save_method_name, _save)
        setattr(klass, load_method_name, _load)
        setattr(klass, init_and_load_classmethod_name, _init_and_load_from)

        return klass

    return _save_load
