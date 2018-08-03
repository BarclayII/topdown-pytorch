import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
import torchvision
import numpy as np
from glimpse import create_glimpse
from util import cuda, area, intersection

def num_nodes(lvl, brch):
    return (brch ** (lvl + 1) - 1) // (brch - 1)

def F_reg_pc(g_par, g_chd):
    """
    Regularization term(Parent-Child)
    """
    chd_area = area(g_chd)
    bbox_penalty = (chd_area - intersection(g_par, g_chd) + 1e-6) / \
                   (chd_area + 1e-6)
    return bbox_penalty.clamp(min=0)

def F_reg_cc(g_chd_list):
    """
    Regularization term(among childs)
    """
    areas = [area(g_chd) for g_chd in g_chd_list]
    chds_penalty = T.zeros_like(areas[0])
    for i, g_chd_i in enumerate(g_chd_list):
        for j, g_chd_j in enumerate(g_chd_list):
            if i < j:
                intersection_area = intersection(g_chd_i, g_chd_j)
                union_area = (areas[i] + areas[j] - intersection_area)
                chds_penalty += (intersection_area + 1e-6) / (union_area + 1e-6)

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
    def __init__(self, filters, kernel_size, final_pool_size, h_dims, n_classes, cnn=None):
        super(WhatModule, self).__init__()
        if cnn is None:
            self.cnn = build_cnn(
                filters=filters,
                kernel_size=kernel_size,
                final_pool_size=final_pool_size
            )
        else:
            self.cnn = cnn
        self.net_h = nn.Sequential(
            nn.Linear(filters[-1] * np.prod(final_pool_size), h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, h_dims),
        )
        self.net_p = nn.Sequential( # net_p is preserved
            nn.ReLU(),
            nn.Linear(h_dims, n_classes),
        )

    def forward(self, glimpse_kxk, readout=True):
        batch_size = glimpse_kxk.shape[0]
        h = self.net_h(self.cnn(glimpse_kxk).view(batch_size, -1))
        return h if not readout else self.net_p(h)


class WhereModule(nn.Module):
    def __init__(self, filters, kernel_size, final_pool_size, h_dims, g_dims, cnn=None):
        super(WhereModule, self).__init__()
        if cnn is None:
            self.cnn = build_cnn(
                filters=filters,
                kernel_size=kernel_size,
                final_pool_size=final_pool_size
            )
        else:
            self.cnn = cnn
        self.net_g = nn.Sequential(
            nn.Linear(filters[-1] * np.prod(final_pool_size), h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, h_dims),
        )
        self.net_p = nn.Sequential(
            nn.ReLU(),
            nn.Linear(h_dims * 2, g_dims),
        )

    def forward(self, glimpse_kxk, readout=True):
        batch_size = glimpse_kxk.shape[0]
        g = self.net_g(self.cnn(glimpse_kxk).view(batch_size, -1))
        return g if not readout else self.net_p(g)


class TreeItem(object):
    def __init__(self, b=None, bbox=None, h=None, y=None, att=None, g=None):
        self.b = b
        self.bbox = bbox
        self.h = h
        self.y = y
        self.att = att
        self.g = g

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

    def forward(self, input):
        return self.net_att(input) if self.net_att is not None else \
                cuda(T.ones(*input.shape[:-1], 1))


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
                 att_type='self',
                 glimpse_type='gaussian',
                 pc_coef=0,
                 cc_coef=0,
                 ):
        super(TreeBuilder, self).__init__()

        glimpse = create_glimpse(glimpse_type, glimpse_size)

        g_dims = glimpse.att_params

        net_phi = nn.ModuleList(
                WhatModule(what_filters, kernel_size, final_pool_size, h_dims,
                           n_classes)
                for _ in range(n_levels + 1)
                )
        net_b = nn.ModuleList(
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.Linear(h_dims * 2, g_dims * n_branches),
                    )
                for _ in range(n_levels + 1)
                )
        net_b_to_h = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(g_dims, h_dims),
                    nn.ReLU(),
                    nn.Linear(h_dims, h_dims),
                    )
                for _ in range(n_levels + 1)
                )

        batch_norm = nn.BatchNorm1d(h_dims * 2)
        self.pc_coef = pc_coef
        self.cc_coef = cc_coef
        self.batch_norm = batch_norm
        self.net_phi = net_phi
        self.net_b = net_b
        self.net_b_to_h = net_b_to_h
        self.net_att = SelfAttentionModule(h_dims * 2, a_dims, att_type)
        self.glimpse = glimpse
        self.n_branches = n_branches
        self.n_levels = n_levels
        self.g_dims = g_dims
        self.reg_type = reg_type

    def noderange(self, level):
        return range(self.n_branches ** level - 1, self.n_branches ** (level + 1) - 1) \
                if self.n_branches > 1 \
                else range(level, level + 1)

    def forward(self, x, lvl=None):
        if lvl is None:
            lvl = self.n_levels

        batch_size, n_channels, n_rows, n_cols = x.shape

        t = [TreeItem() for _ in range(num_nodes(lvl, self.n_branches))]
        # root init
        t[0].b = x.new(batch_size, self.g_dims).zero_()

        loss_pc = 0
        loss_cc = 0
        for l in range(0, lvl + 1):
            current_level = self.noderange(l)

            b = T.stack([t[i].b for i in current_level], 1)
            bbox, _ = self.glimpse.rescale(b, False)
            g = self.glimpse(x, bbox)
            n_glimpses = g.shape[1]
            g_flat = g.view(batch_size * n_glimpses, *g.shape[2:])
            phi = self.net_phi[l](g_flat, readout=False)
            h_b = self.net_b_to_h[l](b.view(batch_size * n_glimpses, self.g_dims))
            h = self.batch_norm(T.cat([phi, h_b], dim=-1))
            att = self.net_att(h).view(batch_size, n_glimpses, -1)

            delta_b = (self.net_b[l](h)
                       .view(batch_size, n_glimpses, self.n_branches, self.g_dims))
            new_b = b[:, :, None] + delta_b

            h = h.view(batch_size, n_glimpses, *h.shape[1:])

            for k, i in enumerate(current_level):
                t[i].bbox = bbox[:, k]
                t[i].g = g[:, k]
                t[i].h = h[:, k]
                t[i].att = att[:, k]

                if l != lvl:
                    for j in range(self.n_branches):
                        t[i * self.n_branches + j + 1].b = new_b[:, k, j]
                if l != 0:
                    loss_pc += F_reg_pc(
                            t[(i - 1) // self.n_branches].bbox,
                            t[i].bbox
                            ).mean()
                    if (k + 1) % self.n_branches == 0:
                        loss_cc += F_reg_cc(
                            [t[current_level[k - j]].bbox for j in range(self.n_branches)]
                            ).mean()

        return t, loss_pc * self.pc_coef + loss_cc * self.cc_coef


class ReadoutModule(nn.Module):
    def __init__(self, h_dims=128, n_classes=10, n_branches=1, n_levels=1):
        super(ReadoutModule, self).__init__()

        self.predictor = nn.Linear(h_dims * 2, n_classes)
        self.n_branches = n_branches
        self.n_levels = n_levels

    def forward(self, t):
        #nodes = t[-self.n_branches ** self.n_levels:]
        results = []
        for lvl in range(self.n_levels + 1):
            nodes = t[:num_nodes(lvl, self.n_branches)]
            att = F.softmax(T.stack([node.att for node in nodes], 1), dim=1)
            h = T.stack([node.h for node in nodes], 1)
            results.append((self.predictor((h * att).sum(dim=1)), att.squeeze(-1)))
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
