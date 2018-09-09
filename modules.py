import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
import torchvision
import torchvision.models
import numpy as np
from glimpse import create_glimpse
from util import cuda, area, intersection
from transform import *
import itertools

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
    margin = 0.25
    area_a = area(chd_a)
    area_b = area(chd_b)
    intersection_area = intersection(chd_a, chd_b)
    chds_penalty = F.relu((intersection_area + 1e-6) / (area_a + 1e-6) - margin) + \
                   F.relu((intersection_area + 1e-6) / (area_b + 1e-6) - margin)
    return chds_penalty

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
            )
        else:
            self.cnn = cnn
            assert "Not implemented yet"
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

class TreeItem(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._attrs = ['b', 'bbox', 'fm', 'y', 'g', 'alpha']

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


class SelfAttentionModule(nn.Module):
    def __init__(self, h_dims, a_dims, att_type=None):
        super(SelfAttentionModule, self).__init__()
        if att_type == 'tanh':
            self.net_att = nn.Sequential(
                nn.Linear(h_dims, a_dims),
                nn.Tanh(),
                nn.Linear(a_dims, 1, bias=False)
            )
        elif att_type == 'naive':
            self.net_att = nn.Sequential(
                nn.Linear(h_dims, 1),
                nn.LeakyReLU()
            )
        else:  # 'mean'
            self.net_att = None

    #@profile
    def forward(self, input):
        if self.net_att is not None:
            return self.net_att(input)
        else:
            return T.ones(*input.shape[:-1], 1, device=input.device)


class TreeBuilder(nn.Module):
    def __init__(self,
                 glimpse_size=(15, 15),
                 what_filters=[16, 32, 64, 128, 256],
                 where_filters=[16, 32],
                 kernel_size=(3, 3),
                 final_pool_size=(5, 5), #(4, 4),
                 h_dims=128,
                 a_dims=50,
                 n_classes=10,
                 n_branches=1,
                 n_levels=1,
                 att_type='self',
                 glimpse_type='gaussian',
                 pc_coef=0,
                 cc_coef=0,
                 res_coef=0,
                 what__cnn=None,
                 what__fix=False,
                 what__in_dims=None,
                 ):
        super(TreeBuilder, self).__init__()

        fm_target_size = (15, 15)
        fm_glim_size = final_pool_size

        glimpse = create_glimpse(glimpse_type, glimpse_size)
        glimpse_fm = create_glimpse(glimpse_type, fm_glim_size)
        g_dims = glimpse.att_params

        net_phi = nn.ModuleList(
                WhatModule(what_filters, kernel_size, final_pool_size, h_dims,
                           n_classes, cnn=what__cnn, fix=what__fix,
                           in_dims=what__in_dims)
                for _ in range(n_levels + 1)
                )

        net_b = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(h_dims * np.prod(fm_target_size), h_dims),
                    nn.ReLU(),
                    nn.Linear(h_dims, h_dims),
                    nn.ReLU(),
                    nn.Linear(h_dims, g_dims * n_branches),
                    )
                for _ in range(n_levels + 1)
                )

        self.pc_coef = pc_coef
        self.cc_coef = cc_coef
        self.res_coef = res_coef
        self.net_phi = net_phi
        self.net_b = net_b
        self.glimpse = glimpse
        self.glimpse_fm = glimpse_fm
        self.n_branches = n_branches
        self.n_levels = n_levels
        self.g_dims = g_dims
        self.h_dims = h_dims
        self.fm_glim_size = fm_glim_size
        self.fm_target_size = fm_target_size

    def noderange(self, level):
        return range(num_nodes(level - 1, self.n_branches), num_nodes(level, self.n_branches)) \
                if self.n_branches > 1 \
                else range(level, level + 1)

    def forward_layer(self, x, l, b):
        batch_size = x.shape[0]
        bbox, _ = self.glimpse.rescale(b, False)
        x_g = self.glimpse(x, bbox)
        n_glimpses = x_g.shape[1]
        x_g_flat = x_g.view(batch_size * n_glimpses, *x_g.shape[2:])
        fm = self.net_phi[l](x_g_flat)
        fm = fm.view(batch_size, n_glimpses, *fm.shape[1:])
        abs_bbox = self.glimpse_fm._to_absolute_attention(bbox, self.fm_target_size)
        fm_new, fm_alpha = F_spatial_feature_map(fm, abs_bbox, self.fm_target_size)

        new_b = (self.net_b[l](fm_new.view(batch_size, n_glimpses, -1))
                 .view(batch_size, n_glimpses, self.n_branches, self.g_dims))
        return bbox, x_g, new_b, fm_new, fm_alpha

    def forward(self, x, lvl=None):
        batch_size, channels, row, col = x.shape
        if lvl is None:
            lvl = self.n_levels

        t = [TreeItem() for _ in range(num_nodes(lvl, self.n_branches))]
        # root init
        t[0].b = x.new(batch_size, self.g_dims).zero_()

        loss_pc = 0
        loss_cc = 0
        loss_res = 0
        pc_par = []
        pc_chd = []
        cc_chd_a = []
        cc_chd_b = []

        for l in range(0, lvl + 1):
            current_level = self.noderange(l)
            b = T.stack([t[i].b for i in current_level], 1)
            bbox, g, new_b, fm, alpha = self.forward_layer(x, l, b)
            # propagate
            for k, i in enumerate(current_level):
                t[i].bbox = bbox[:, k]
                if l == lvl:
                    k_x, k_y = self.glimpse.glim_size
                    loss_res += (1. / len(current_level)) *\
                        F_reg_res(t[i].bbox, row, col, k_x, k_y)

                t[i].g = g[:, k]
                t[i].fm = fm[:, k]
                t[i].alpha = alpha[:, k]

                if l != lvl:
                    for j in range(self.n_branches):
                        t[i * self.n_branches + j + 1].b = new_b[:, k, j]
                if l != 0:
                    pc_par.append(t[(i - 1) // self.n_branches].bbox)
                    pc_chd.append(t[i].bbox)
                    if (k + 1) % self.n_branches == 0:
                        for i, j in itertools.combinations(range(k - self.n_branches + 1, k + 1), 2):
                            cc_chd_a.append(t[current_level[i]].bbox)
                            cc_chd_b.append(t[current_level[j]].bbox)

        loss_pc = F_reg_pc(T.stack(pc_par, 1), T.stack(pc_chd, 1)) if lvl >= 1 else x.new(1).zero_()
        loss_cc = F_reg_cc(T.stack(cc_chd_a, 1), T.stack(cc_chd_b, 1)) if lvl >= 1 and self.n_branches > 1 else x.new(1).zero_()

        return t, (loss_pc * self.pc_coef, loss_cc * self.cc_coef, loss_res * self.res_coef)

class HomoscedasticModule(nn.Module):
    def __init__(self, n_levels):
        super(HomoscedasticModule, self).__init__()
        self.coef_lambda = nn.Parameter(T.ones(n_levels), requires_grad=True)

class ReadoutModule(nn.Module):
    def __init__(self, h_dims=128, g_dims=6, n_classes=10, n_branches=1, n_levels=1):
        super(ReadoutModule, self).__init__()
        fm_target_size = (15, 15)
        self.predictor = nn.ModuleList(
            nn.Linear(h_dims, n_classes)
            for _ in range(n_levels + 1)
        )
        self.n_branches = n_branches
        self.n_levels = n_levels
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, t, lvls=None):
        if lvls is None:
            lvls = self.n_levels
        results = []
        hs = []
        for lvl in range(lvls + 1):
            def to_detach(x, idx):
                if idx >= num_nodes(lvl - 1, self.n_branches):
                    return x
                else:
                    return x.detach()

            nodes = t[:num_nodes(lvl, self.n_branches)]
            accum_fm = 0
            for idx, node in enumerate(nodes):
                accum_fm = accum_fm * (1 - node.alpha) +\
                    to_detach(node.fm, idx) * node.alpha
            h = self.avgpool(accum_fm)
            h = h.view(h.shape[0], -1)
            results.append(self.predictor[lvl](h))
            hs.append(h)

        self.hs = hs
        return results

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
