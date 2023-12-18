from typing import Tuple
from torch.optim import AdamW, Adam

# optimizer

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, no_wd_params

def get_adam_optimizer(
    params,
    lr: float = 1e-4,
    wd: float = 1e-2,
    betas: Tuple[int, int] = (0.9, 0.99),
    eps: float = 1e-8,
    filter_by_requires_grad = False,
    omit_gammas_and_betas_from_wd = True,
    **kwargs
):
    has_weight_decay = wd > 0.

    if filter_by_requires_grad:
        params = [t for t in params if t.requires_grad]

    opt_kwargs = dict(
        lr = lr,
        betas = betas,
        eps = eps
    )

    if not has_weight_decay:
        return Adam(params, **opt_kwargs)

    opt_kwargs = {'weight_decay': wd, **opt_kwargs}

    if not omit_gammas_and_betas_from_wd:
        return AdamW(params, **opt_kwargs)

    # there is an early practice where betas and gammas are omitted from weight decay in transformers
    # unsure whether it is really needed or not

    wd_params, no_wd_params = separate_weight_decayable_params(params)

    params = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(params, **opt_kwargs)
