import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
from glimpse import inverse_gaussian_masks

def F_spatial_feature_map(x, g, target_fm_shape):
    """
    x: batch_size, n_channels, rows, cols
    g: batch_size, 1, att_params (cx, cy, dx, dy, sx, sy)
    For bird dataset, we set target_fm_shape to (16, 16).

    return:
    (batch_size, nchannels, target_fm_shape)
    """
    batch_size, n_channels, rows, cols = x.shape
    cx, cy, dx, dy, sx, sy = T.unbind(g, -1)
    target_rows, target_cols = target_fm_shape

    # (batch_size, rows, target_rows)
    Fy = inverse_gaussian_masks(cy, dy, sy, rows, target_rows)
    # (batch_size, cols, target_cols)
    Fx = inverse_gaussian_masks(cx, dx, sx, cols, target_cols)

    # (batch_size, 1, rows, target_rows)
    Fy = Fy.unsqueeze(1)
    # (batch_size, 1, cols, target_cols)
    Fx = Fx.unsqueeze(1)

    spatial_fm = Fy.transpose(-1, -2) @ x @ Fx
    return spatial_fm
