from typing import Tuple
import torch.nn.functional as F

def exists(v):
    return v is not None

# padding and slicing

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

# masking related

def masked_mean(tensor, mask, dim = -1, eps = 1e-5):
    if not exists(mask):
        return tensor.mean(dim = dim)

    tensor.masked_fill_(~mask, 0.)

    total_el = mask.sum(dim = dim)
    num = tensor.sum(dim = dim)
    den = total_el.float().clamp(min = eps)
    mean = num / den
    mean.masked_fill_(total_el == 0, 0.)
    return mean

def maybe_and_mask(*masks):
    masks = [*filter(exists, masks)]
    if len(masks) == 0:
        return None

    mask, *rest_masks = masks
    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask
