import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
import numpy as np
import argparse
import os
import sys
from glimpse import create_glimpse
from util import cuda
from datasets import MNISTMulti
from viz import VisdomWindowManager
from stats.utils import *
import tqdm

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
        if i < len(filters) - 1:
            cnn_list.append(nn.LeakyReLU())
    cnn_list.append(nn.AdaptiveMaxPool2d(final_pool_size))

    return nn.Sequential(*cnn_list)

def data_generator(dataset, batch_size, shuffle):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    for _x, _y, _B in dataloader:
        x = _x[:, None].expand(_x.shape[0], 3, _x.shape[1], _x.shape[2]).float() / 255.
        y = _y.squeeze(1)
        b = _B.squeeze(1).float() / 200
        yield cuda(x), cuda(y), cuda(b)

class WhatModule(nn.Module):
    def __init__(self, filters, kernel_size, final_pool_size, h_dims, n_classes):
        super(WhatModule, self).__init__()
        self.cnn = build_cnn(
            filters=filters,
            kernel_size=kernel_size,
            final_pool_size=final_pool_size
        )
        self.net_h = nn.Sequential(
            nn.Linear(filters[-1] * np.prod(final_pool_size), h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, h_dims),
        )
        self.net_p = nn.Sequential(
            nn.ReLU(),
            nn.Linear(h_dims, n_classes),
        )

    def forward(self, glimpse_kxk, readout=True):
        batch_size = glimpse_kxk.shape[0]
        h = self.net_h(self.cnn(glimpse_kxk).view(batch_size, -1))
        return h if not readout else self.net_p(h)

class WhereModule(nn.Module):
    def __init__(self, filters, kernel_size, final_pool_size, h_dims, g_dims):
        super(WhereModule, self).__init__()
        self.cnn = build_cnn(
            filters=filters,
            kernel_size=kernel_size,
            final_pool_size=final_pool_size
        )
        self.net_g = nn.Sequential(
            nn.Linear(filters[-1] * np.prod(final_pool_size), h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, h_dims),
        )
        self.net_p = nn.Sequential(
            nn.ReLU(),
            nn.Linear(h_dims, g_dims),
        )

    def forward(self, glimpse_kxk, readout=True):
        batch_size = glimpse_kxk.shape[0]
        g = self.net_g(self.cnn(glimpse_kxk).view(batch_size, -1))
        return g if not readout else self.net_p(g)


class TreeItem(object):
    def __init__(self, b=None, bbox=None, h_where=None, h_what=None, y=None, g=None):
        self.b = b
        self.bbox = bbox
        self.h_where = h_where
        self.h_what = h_what
        self.y = y
        self.g = g


class TreeBuilder(nn.Module):
    def __init__(self,
                 glimpse_size=(15, 15),
                 what_filters=[16, 32, 64, 128, 256],
                 where_filters=[16, 32],
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 h_dims=128,
                 n_classes=10,
                 n_branches=2,
                 n_levels=2,
                 ):
        super(TreeBuilder, self).__init__()

        glimpse = create_glimpse('gaussian', glimpse_size)

        g_dims = glimpse.att_params

        net_phi = WhatModule(what_filters, kernel_size, final_pool_size, h_dims, n_classes)
        net_g = WhereModule(where_filters, kernel_size, final_pool_size, h_dims, g_dims)
        net_b = nn.Sequential(
                nn.ReLU(),
                nn.Linear(h_dims, g_dims * n_branches),
                )
        net_phi = cuda(net_phi)
        net_g = cuda(net_g)

        self.net_phi = net_phi
        self.net_g = net_g
        self.net_b = net_b
        self.glimpse = glimpse
        self.n_branches = 2
        self.n_levels = 2
        self.g_dims = g_dims

    @property
    def n_nodes(self):
        return self.n_branches ** (self.n_levels + 1) - 1

    @property
    def n_internals(self):
        return self.n_branches ** self.n_levels - 1

    def forward(self, x):
        batch_size, n_channels, n_rows, n_cols = x.shape

        t = [TreeItem() for _ in range(self.n_nodes)]
        
        # root init
        t[0].b = x.new(batch_size, self.g_dims).zero_()

        for l in range(0, self.n_levels + 1):
            current_level = range(self.n_branches ** l - 1,
                                  self.n_branches ** (l + 1) - 1)
            b = T.stack([t[i].b for i in current_level], 1)
            bbox, _ = self.glimpse.rescale(b, False)
            g = self.glimpse(x, bbox)
            n_glimpses = g.shape[1]
            g_flat = g.view(batch_size * n_glimpses, *g.shape[2:])
            h_where = self.net_g(g_flat, readout=False)
            delta_b = (self.net_b(h_where)
                       .view(batch_size, n_glimpses, self.n_branches, self.g_dims))
            new_b = b[:, :, None] + delta_b
            h_where = h_where.view(batch_size, n_glimpses, *h_where.shape[1:])
            h_what = self.net_phi(g_flat, readout=False)
            y = self.net_phi.net_p(h_what).view(batch_size, n_glimpses, -1)
            h_what = h_what.view(batch_size, n_glimpses, *h_what.shape[1:])

            for k, i in enumerate(current_level):
                t[i].bbox = bbox[:, k]
                t[i].g = g[:, k]
                t[i].h_where = h_where[:, k]
                t[i].h_what = h_what[:, k]
                t[i].y = y[:, k]

                if l != self.n_levels:
                    for j in range(self.n_branches):
                        t[i * self.n_branches + j + 1].b = new_b[:, k, j]

        return t


class ReadoutModule(nn.Module):
    def __init__(self, h_dims=128, n_classes=10, n_branches=2, n_levels=2):
        super(ReadoutModule, self).__init__()

        self.predictor = nn.Linear(h_dims, n_classes)
        self.n_branches = n_branches
        self.n_levels = n_levels

    def forward(self, t):
        leaves = t[-self.n_branches ** self.n_levels:]
        h_what = T.stack([node.h_what for node in leaves], 1).mean(1)
        return self.predictor(h_what)

parser = argparse.ArgumentParser(description='Alternative')
parser.add_argument('--pretrain', action='store_true', help='pretrain or not pretrain')
parser.add_argument('--row', default=200, type=int, help='image rows')
parser.add_argument('--col', default=200, type=int, help='image cols')
parser.add_argument('--alter', action='store_true', help='indicates whether to use alternative training or joint training')
parser.add_argument('--n', default=5, type=int, help='number of epochs')
parser.add_argument('--log_interval', default=10)
args = parser.parse_args()

mnist_train = MNISTMulti('.', n_digits=1, backrand=0, image_rows=args.row, image_cols=args.col, download=True)
mnist_valid = MNISTMulti('.', n_digits=1, backrand=0, image_rows=args.row, image_cols=args.col, download=False, mode='valid')

builder = cuda(TreeBuilder())
readout = cuda(ReadoutModule())

train_shuffle = True

if args.pretrain:
    pass
else:
    wm = VisdomWindowManager(port=11111)
    n_epochs = args.n
    len_train = len(mnist_train)
    len_valid = len(mnist_valid)
    phase = 'What'
    params = list(builder.parameters()) + list(readout.parameters())
    opt = T.optim.RMSprop(params, lr=3e-5)
    for epoch in range(2 * n_epochs):
        print("Epoch {} starts...".format(epoch))

        batch_size = 64
        train_loader = data_generator(mnist_train, batch_size, train_shuffle)

        sum_loss = 0
        n_batches = len_train // batch_size
        hit = 0
        cnt = 0
        for i, (x, y, b) in enumerate(train_loader):
            t = builder(x)
            y_pred = readout(t)
            loss = F.cross_entropy(
                y_pred, y
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 0.01)
            opt.step()
            sum_loss += loss.item()
            hit += (y_pred.max(dim=-1)[1] == y).sum().item()
            cnt += batch_size
            if i % args.log_interval == 0 and i > 0:
                avg_loss = sum_loss / args.log_interval
                sum_loss = 0
                print('{} phase, Batch {}/{}, loss = {}, acc={}'.format(phase, i, n_batches, avg_loss, hit * 1.0 / cnt))
                hit = 0
                cnt = 0

        batch_size = 256
        valid_loader = data_generator(mnist_valid, batch_size, False)
        cnt = 0
        hit = 0
        sum_loss = 0
        with T.no_grad():
            for i, (x, y, b) in enumerate(valid_loader):
                t = builder(x)
                y_pred = readout(t)
                loss = F.cross_entropy(
                    y_pred, y
                )
                sum_loss += loss.item()
                hit += (y_pred.max(dim=-1)[1] == y).sum().item()
                cnt += batch_size
                #if i == 0:
                #    sample_imgs = x[:10]
                #    sample_gs = g[:10, 0, :]
                #    sample_bboxs = glimpse_to_xyhw(sample_gs) * 200
                #    statplot = StatPlot(5, 2)
                #    for j in range(10):
                #        statplot.add_image(sample_imgs[j][0], bboxs=[sample_bboxs[j]])
                #    wm.display_mpl_figure(statplot.fig)
                #    wm.append_mpl_figure_to_sequence('bbox', statplot.fig)

            avg_loss = sum_loss / i
        print("Loss on valid set: {}".format(avg_loss))
        print("Accuracy on valid set: {}".format(hit * 1.0 / cnt))
        #wm.append_scalar('loss', avg_loss)
        #wm.append_scalar('acc', hit * 1.0 / cnt)
    #wm.display_mpl_figure_sequence('bbox')
