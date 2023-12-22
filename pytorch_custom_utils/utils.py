from typing import Tuple
import torch.nn.functional as F

def pad_at_dim(t, pad: Tuple[int, int], *, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

def pad_or_slice_to(t, length, *, dim, pad_value = 0):
    curr_length = t.shape[dim]

    if curr_length < length:
        t = pad_at_dim(t, (0, length - curr_length), dim = dim, value = pad_value)
    elif curr_length > length:
        t = slice_at_dim(t, slice(0, length), dim = dim)

    return t
