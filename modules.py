import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
import torchvision
import torchvision.models
import numpy as np
from glimpse import create_glimpse
from util import cuda, area, intersection, list_index_select
from transform import *
import itertools
import random

def F_cauchy(ratio):
    return T.log(1 + T.abs(ratio - 1))

def kl_temperature(y, lbl, temperature=0.01):
    batch_size = y.shape[0]
    n_classes = y.shape[1]
    y_logit = cuda(T.zeros(batch_size, n_classes))
    y_logit.scatter_(1, lbl.unsqueeze(-1), 1)
    return F.kl_div(F.log_softmax(y), F.softmax(y_logit / temperature), size_average=False) / batch_size

def num_nodes(lvl, brch):
    if brch == 1:
        return lvl + 1
    else:
        return (brch ** (lvl + 1) - 1) // (brch - 1)

def F_ent(dist, eps=1e-8):
    """
    Entropy of Distributions.
    Input format:
    (*, C)
    Output format:
    (*)
    """
    return -(dist * T.log(dist + eps)).sum(dim=-1)

def F_reg_res(bbox, row, col, k_x, k_y):
    d_x = (bbox[:, 2] * col) / k_x
    d_y = (bbox[:, 3] * row) / k_y
    s_x = (bbox[:, 4] * col * 2) / k_x
    s_y = (bbox[:, 5] * row * 2) / k_y
    return (F_cauchy(d_x) +
        F_cauchy(d_y) +
        F_cauchy(s_x) +
        F_cauchy(s_y)
        ) / 4.

def F_reg_pc(par, chd):
    """
    Regularization term(Parent-Child)
    """
    chd_area = area(chd)
    bbox_penalty = (chd_area - intersection(par, chd) + 1e-6) / \
                   (chd_area + 1e-6)
    return bbox_penalty.clamp(min=0)

def F_reg_cc(chd_a, chd_b):
    """
    Regularization term(among childs)
    """
    margin = 0
    area_a = area(chd_a)
    area_b = area(chd_b)
    intersection_area = intersection(chd_a, chd_b)
    chds_penalty = F.relu((intersection_area + 1e-6) / (area_a + 1e-6) - margin) + \
                   F.relu((intersection_area + 1e-6) / (area_b + 1e-6) - margin)
    return chds_penalty

def noderange(n_branches, level):
    return range(num_nodes(level - 1, n_branches), num_nodes(level, n_branches)) \
            if n_branches > 1 \
            else range(level, level + 1)


def build_cnn(**config):
    cnn_list = []
    filters = config['filters']
    kernel_size = config['kernel_size']
    in_channels = config.get('in_channels', 3)
    final_pool_size = config['final_pool_size']

    for i in range(len(filters)):
        module = nn.Conv2d(
            in_channels if i == 0 else filters[i-1],
            filters[i],
            kernel_size,
            padding=tuple((_ - 1) // 2 for _ in kernel_size),
            )
        INIT.xavier_uniform_(module.weight)
        INIT.constant_(module.bias, 0)
        cnn_list.append(module)
        cnn_list.append(nn.BatchNorm2d(filters[i]))
        if i < len(filters) - 1:
            cnn_list.append(nn.LeakyReLU())
    cnn_list.append(nn.AdaptiveMaxPool2d(final_pool_size))

    return nn.Sequential(*cnn_list)

class WhatModule(nn.Module):
    def __init__(self, filters, kernel_size, final_pool_size, h_dims, n_classes, cnn=None, in_dims=None, fix=False):
        super(WhatModule, self).__init__()
        if cnn is None:
            self.cnn = build_cnn(
                filters=filters,
                kernel_size=kernel_size,
                final_pool_size=final_pool_size
            )
            in_dims_0 = filters[-1]
        elif isinstance(cnn, str) and cnn.startswith('resnet'):
            cnn = getattr(torchvision.models, cnn)(pretrained=True)
            in_dims_0 = cnn.fc.in_features
            self.cnn = nn.Sequential(
                    cnn.conv1,
                    cnn.bn1,
                    cnn.relu,
                    cnn.maxpool,
                    cnn.layer1,
                    cnn.layer2,
                    cnn.layer3,
                    cnn.layer4,
                    nn.AdaptiveMaxPool2d(final_pool_size)
            )
        else:
            self.cnn = cnn
            raise NotImplementedError
        self.conv1x1 = nn.Conv2d(in_dims_0, h_dims, kernel_size=(1, 1))
        self.fix = fix

    def forward(self, glimpse_kxk):
        batch_size = glimpse_kxk.shape[0]
        if self.fix:
            with T.no_grad():
                fm = self.conv1x1(self.cnn(glimpse_kxk))
        else:
            fm = self.conv1x1(self.cnn(glimpse_kxk))
        return fm


class InverseGlimpse(nn.Module):
    def __init__(self, glimpse_fm, fm_target_size):
        super(InverseGlimpse, self).__init__()
        self.glimpse_fm = glimpse_fm
        self.fm_target_size = fm_target_size

    def forward(self, fm, b):
        abs_b = self.glimpse_fm._to_absolute_attention(b, self.fm_target_size)
        fm_new, fm_alpha = F_spatial_feature_map(fm, abs_b, self.fm_target_size)
        return fm_new, fm_alpha

class GlimpseUpdater(nn.Module):
    def __init__(self, glimpse, input_dims, h_dims, g_dims, n_branches, n_levels):
        super(GlimpseUpdater, self).__init__()
        self.glimpse = glimpse
        net_b = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(input_dims, h_dims),
                    nn.ReLU(),
                    nn.Linear(h_dims, h_dims),
                    nn.ReLU(),
                    nn.Linear(h_dims, g_dims * n_branches),
                    )
                for _ in range(n_levels + 1)
                )
        self.net_b = net_b
        self.n_branches = n_branches
        self.g_dims = g_dims

    def forward(self, b, h, l):
        #fm, alpha = h
        fm = h
        batch_size, n_glimpses = fm.shape[:2]
        delta_b = self.net_b[l](
                fm.detach().view(batch_size, n_glimpses, -1)
        ).view(batch_size, n_glimpses, self.n_branches, self.g_dims)
        delta_b = self.glimpse.rescale(delta_b)
        new_b = self.glimpse.upd_b(b.unsqueeze(2).repeat(1, 1, self.n_branches, 1), delta_b)
        return new_b


class TreeItem(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._attrs = ['b', 'h', 'y', 'g', 'alpha']

    def __getattr__(self, name):
        if not name.startswith('_') and name in self._attrs:
            return self[name]
        return dict.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if not name.startswith('_') and name in self._attrs:
            self[name] = value
        else:
            dict.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name not in self._attrs:
            dict.__delattr__(self, name)


class Regularizer(nn.Module):
    def __init__(self, coef, owner):
        super(Regularizer, self).__init__()
        self.coef = coef
        self.owner = [owner]
        self.n_branches = owner.n_branches
        self.n_levels = owner.n_levels


class PCRegularizer(Regularizer):
    def forward(self, t, row, col, lvl=None):
        """
        row and col is not required here. (unified interface with resolution regularizer)
        """
        if lvl is None:
            lvl = self.n_levels

        pc_par = []
        pc_chd = []

        for l in range(1, lvl + 1):
            current_level = noderange(self.n_branches, l)
            for k, i in enumerate(current_level):
                pc_par.append(t[(i - 1) // self.n_branches].b)
                pc_chd.append(t[i].b)

        loss_pc = F_reg_pc(T.stack(pc_par, 1), T.stack(pc_chd, 1)) if lvl >= 1 else x.new(1).zero_()
        return self.coef * loss_pc


class CCRegularizer(Regularizer):
    def forward(self, t, row, col, lvl=None):
        """
        row and col is not required here. (unified interface with resolution regularizer)
        """
        if lvl is None:
            lvl = self.n_levels

        cc_chd_a = []
        cc_chd_b = []

        for l in range(1, lvl + 1):
            current_level = noderange(self.n_branches, l)
            for k, i in enumerate(current_level):
                if (k + 1) % self.n_branches == 0:
                    for i, j in itertools.combinations(range(k - self.n_branches + 1, k + 1), 2):
                        cc_chd_a.append(t[current_level[i]].b)
                        cc_chd_b.append(t[current_level[j]].b)

        loss_cc = F_reg_cc(T.stack(cc_chd_a, 1), T.stack(cc_chd_b, 1)) if lvl >= 1 and self.n_branches > 1 else cuda(T.zeros(1)) #x.new(1).zero_()
        return self.coef * loss_cc


class ResRegularizer(Regularizer):
    def forward(self, t, row, col, lvl=None):
        if lvl is None:
            lvl = self.n_levels

        loss_res = 0
        current_level = noderange(self.n_branches, lvl)
        k_x, k_y = self.owner[0].glimpse.glim_size
        for k, i in enumerate(current_level):
            loss_res += (1. / len(current_level)) * F_reg_res(t[i].b, row, col, k_x, k_y)

        return self.coef * loss_res


class TreeBuilder(nn.Module):
    def __init__(self,
                 glimpse_size=(15, 15),
                 what_filters=[16, 32, 64, 128, 256],
                 where_filters=[16, 32],
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 h_dims=128,
                 a_dims=50,
                 n_classes=10,
                 n_branches=1,
                 n_levels=1,
                 glimpse_type='gaussian',
                 explore=False,
                 bind=False,
                 regularizer_classes={
                     PCRegularizer: 0,
                     CCRegularizer: 0,
                     ResRegularizer: 0,
                     },
                 what__cnn=None,
                 what__fix=False,
                 what__in_dims=None,
                 ):
        super(TreeBuilder, self).__init__()

        fm_target_size = (10, 10) #(16, 16)
        fm_glim_size = final_pool_size

        glimpse = create_glimpse(glimpse_type, glimpse_size, explore=explore, bind=bind)
        glimpse_fm = create_glimpse(glimpse_type, fm_glim_size)
        g_dims = glimpse.att_params

        net_phi = nn.ModuleList(
                WhatModule(what_filters, kernel_size, final_pool_size, h_dims,
                           n_classes, cnn=what__cnn, fix=what__fix,
                           in_dims=what__in_dims)
                for _ in range(n_levels + 1)
                )

        net_h = InverseGlimpse(glimpse_fm, fm_target_size)
        upd_b = GlimpseUpdater(
                glimpse,
                h_dims * np.prod(final_pool_size),
                h_dims,
                g_dims,
                n_branches,
                n_levels
                )

        self.net_phi = net_phi
        self.net_h = net_h
        self.upd_b = upd_b
        self.glimpse = glimpse
        self.glimpse_fm = glimpse_fm
        self.n_branches = n_branches
        self.n_levels = n_levels
        self.g_dims = g_dims
        self.h_dims = h_dims
        self.fm_glim_size = fm_glim_size
        self.fm_target_size = fm_target_size

        self.regs = nn.ModuleList()
        for reg_cls, reg_coef in regularizer_classes.items():
            self.regs.append(reg_cls(reg_coef, self))

    '''
    def forward_layer(self, x, l, b):
        batch_size = x.shape[0]
        x_g = self.glimpse(x, b)
        n_glimpses = x_g.shape[1]
        x_g_flat = x_g.view(batch_size * n_glimpses, *x_g.shape[2:])
        fm = self.net_phi[l](x_g_flat)
        fm = fm.view(batch_size, n_glimpses, *fm.shape[1:])
        abs_b = self.glimpse_fm._to_absolute_attention(b, self.fm_target_size)
        fm_new, fm_alpha = F_spatial_feature_map(fm, abs_b, self.fm_target_size)
        delta_b = self.net_b[l](
                fm.detach().view(batch_size, n_glimpses, -1)
        ).view(batch_size, n_glimpses, self.n_branches, self.g_dims)
        delta_b = self.glimpse.rescale(delta_b)
        new_b = self.glimpse.upd_b(b.unsqueeze(2).repeat(1, 1, self.n_branches, 1), delta_b)
        return x_g, new_b, fm_new, fm_alpha
    '''

    def forward_layer(self, x, l, b):
        batch_size = x.shape[0]
        x_g = self.glimpse(x, b)
        n_glimpses = x_g.shape[1]
        x_g_flat = x_g.view(batch_size * n_glimpses, *x_g.shape[2:])
        fm = self.net_phi[l](x_g_flat)
        fm = fm.view(batch_size, n_glimpses, *fm.shape[1:])
        h = self.net_h(fm, b)
        new_b = None
        if l < self.n_levels:
            new_b = self.upd_b(b, fm, l)
        return x_g, new_b, h

    def forward(self, x, lvl=None):
        batch_size, channels, row, col = x.shape
        if lvl is None:
            lvl = self.n_levels

        t = [TreeItem() for _ in range(num_nodes(lvl, self.n_branches))]
        # root init
        t[0].b = x.new(batch_size, self.g_dims).zero_()
        t[0].b[:, :2] = 0.5
        t[0].b[:, 2:4] = 1
        t[0].b[:, 4:] = 0.5

        for l in range(0, lvl + 1):
            current_level = noderange(self.n_branches, l)
            b = T.stack([t[i].b for i in current_level], 1)
            g, new_b, h = self.forward_layer(x, l, b)
            # propagate
            for k, i in enumerate(current_level):
                t[i].g = list_index_select(g, (slice(None), k))
                # If we are using inverse glimpse, each h contains a tuple (fm, alpha)
                t[i].h = list_index_select(h, (slice(None), k))

                if l != lvl:
                    for j in range(self.n_branches):
                        t[i * self.n_branches + j + 1].b = list_index_select(
                                new_b, (slice(None), k, j))

        regularizer_losses = [r(t, row, col, lvl) for r in self.regs]

        return t, regularizer_losses

class AlphaChannelReadoutModule(nn.Module):
    '''
    Only works when using inverse glimpse (i.e. have fm and alpha channels)
    '''
    def __init__(self, h_dims=128, g_dims=6, n_classes=10, n_branches=1, n_levels=1):
        super(ReadoutModule, self).__init__()
        self.predictor = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(h_dims, h_dims),
                    nn.ReLU(),
                    nn.Linear(h_dims, n_classes)
                ) for _ in range(n_levels + 1)
            )
        self.n_branches = n_branches
        self.n_levels = n_levels
        self.avgpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, t, lvls=None):
        if lvls is None:
            lvls = self.n_levels
        results = []
        hs = []
        accum_fm = 0
        for lvl in range(lvls + 1):
            nodes = t[num_nodes(lvl - 1, self.n_branches): num_nodes(lvl, self.n_branches)]
            random.shuffle(nodes)
            for node in nodes:
                fm, alpha = node.h
                accum_fm = accum_fm * (1 - alpha) + fm * alpha
            h = self.avgpool(accum_fm)
            h = h.view(h.shape[0], -1)
            results.append(self.predictor[lvl](h))
            hs.append(h)
            accum_fm.detach_()

        self.hs = hs
        return results


class NodewiseMaxPoolingReadoutModule(nn.Module):
    '''
    Only works when h is a hidden state tensor
    '''
    def __init__(self, h_dims=128, g_dims=6, n_classes=10, n_branches=1, n_levels=1):
        super(ReadoutModule, self).__init__()
        self.predictor = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(h_dims, h_dims),
                    nn.ReLU(),
                    nn.Linear(h_dims, n_classes)
                ) for _ in range(n_levels + 1)
            )
        self.n_branches = n_branches
        self.n_levels = n_levels

    def forward(self, t, lvls=None):
        if lvls is None:
            lvls = self.n_levels

        results = []
        hs = []

        for lvl in range(lvls + 1):
            # Includes all nodes before the @lvl'th level
            nodes = t[:num_nodes(lvl, self.n_branches)]
            h, _ = T.stack([node.h for node in nodes], 1).max(1)
            results.append(self.predictor[lvl](h))
            hs.append(h)

        self.hs = hs
        return results


"""
class MultiscaleGlimpse(nn.Module):
    multiplier = cuda(T.FloatTensor(
        [#[1, 1, 0.5, 0.5, 0.5, 0.5],
             [1, 1, 1, 1, 1, 1],
             #[1, 1, 1.5, 1.5, 1.5, 1.5],
             ]
            ))

    def __init__(self, **config):
        nn.Module.__init__(self)

        glimpse_type = config['glimpse_type']
        self.glimpse_size = config['glimpse_size']
        self.n_glimpses = config['n_glimpses']
        self.glimpse = create_glimpse(glimpse_type, self.glimpse_size)

    def forward(self, x, b=None, flatten_glimpses=True):
        batch_size, n_channels = x.shape[:2]
        if b is None:
            # defaults to full canvas
            b = x.new(batch_size, self.glimpse.att_params).zero_()
        b, _ = self.glimpse.rescale(b[:, None], False)
        b = b.repeat(1, self.n_glimpses, 1) * self.multiplier[None, :, :self.glimpse.att_params]
        g = self.glimpse(x, b)
        if flatten_glimpses:
            g = g.view(
                batch_size * self.n_glimpses, n_channels, self.glimpse_size[0], self.glimpse_size[1])
        else:
            g = g.view(
                batch_size, self.n_glimpses, n_channels, self.glimpse_size[0], self.glimpse_size[1])
        return g
"""
