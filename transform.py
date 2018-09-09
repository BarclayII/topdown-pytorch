import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
from glimpse import inverse_gaussian_masks_surrogate

def F_spatial_feature_map(x, absolute_att, target_fm_shape):
    """
    input:
        x: (batch_size, n_glims, n_channels, rows, cols)
        absolute_att: (batch_size, n_glims, att_params)
        att_params: cx, cy, dx, dy, sx, sy
        target_fm_shape: (16, 16) for bird dataset
    return:
        (batch_size, n_channels, target_fm_shape)
    """
    batch_size, n_glims, n_channels, rows, cols = x.shape
    target_rows, target_cols = target_fm_shape
    cx, cy, dx, dy, sx, sy = T.unbind(absolute_att, -1)

   # with T.no_grad():
    # (batch_size, n_glims, rows, target_rows)
    Fy_inv = inverse_gaussian_masks_surrogate(cy, dy, sy, rows, target_rows).detach()
    # (batch_size, n_glims, cols, target_cols)
    Fx_inv = inverse_gaussian_masks_surrogate(cx, dx, sx, cols, target_cols).detach()

    # (batch_size, n_glims, 1, rows, target_rows)
    Fy_inv = Fy_inv.unsqueeze(2)
    assert Fy_inv.shape == (batch_size, n_glims, 1, rows, target_rows), ( (batch_size, n_glims, 1, rows, target_rows), Fy_inv.shape)
    # (batch_size, n_glims, 1, cols, target_cols)
    Fx_inv = Fx_inv.unsqueeze(2)
    assert Fy_inv.shape == (batch_size, n_glims, 1, rows, target_rows), Fy_inv.shape

    # (batch_size, n_glims, n_channels, rows, cols)
    x = T.cat([x, x.new(batch_size, n_glims, 1, rows, cols).fill_(1)], 2)
    assert x.shape == (batch_size, n_glims, n_channels + 1, rows, cols), x.shape

    # (batch_size, n_glims, n_channels, target_rows, target_cols)
    spatial_fm = Fy_inv.transpose(-1, -2) @ x @ Fx_inv
    return spatial_fm[:, :, :-1, ...], spatial_fm[:, :, -1:, ...]
