import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
import torchvision
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

def num_nodes(lvl, brch):
    return (brch ** (lvl + 1) - 1) // (brch - 1)

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
            nn.Linear(h_dims * 2, n_classes),
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


class TreeBuilder(nn.Module):
    def __init__(self,
                 glimpse_size=(15, 15),
                 what_filters=[16, 32, 64, 128, 256],
                 where_filters=[16, 32],
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 h_dims=128,
                 n_classes=10,
                 n_branches=1,
                 n_levels=1,
                 ):
        super(TreeBuilder, self).__init__()

        glimpse = create_glimpse('gaussian', glimpse_size)

        g_dims = glimpse.att_params

        net_phi = nn.ModuleList(
                WhatModule(what_filters, kernel_size, final_pool_size, h_dims,
                           n_classes)
                for _ in range(n_levels + 1)
                )
        net_g = nn.ModuleList(
                WhereModule(where_filters, kernel_size, final_pool_size, h_dims,
                            g_dims)
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
        net_att = nn.Sequential(
            nn.Linear(h_dims * 2, 1),
            nn.LeakyReLU(),
        )
        batch_norm = nn.BatchNorm1d(h_dims * 2)
        self.batch_norm = batch_norm
        self.net_phi = net_phi
        self.net_g = net_g
        self.net_b = net_b
        self.net_b_to_h = net_b_to_h
        self.net_att = net_att
        self.glimpse = glimpse
        self.n_branches = n_branches
        self.n_levels = n_levels
        self.g_dims = g_dims

    @property
    def n_nodes(self):
        return (self.n_branches ** (self.n_levels + 1) - 1 if self.n_branches > 1
                else self.n_levels + 1)

    def noderange(self, level):
        return range(self.n_branches ** level - 1, self.n_branches ** (level + 1) - 1) \
                if self.n_branches > 1 \
                else range(level, level + 1)

    def forward(self, x, lvl):
        batch_size, n_channels, n_rows, n_cols = x.shape

        t = [TreeItem() for _ in range(num_nodes(lvl, self.n_branches))]
        # root init
        t[0].b = x.new(batch_size, self.g_dims).zero_()

        for l in range(0, lvl + 1):
            current_level = self.noderange(l)

            b = T.stack([t[i].b for i in current_level], 1)
            bbox, _ = self.glimpse.rescale(b, False)
            g = self.glimpse(x, bbox)
            n_glimpses = g.shape[1]
            g_flat = g.view(batch_size * n_glimpses, *g.shape[2:])
            phi = self.net_g[l](g_flat, readout=False)
            #self.batch_norm(T.cat([self.net_g[l](g_flat, readout=False),
            #                       b.view(-1, self.g_dims)], dim=-1))
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

        return t


class ReadoutModule(nn.Module):
    def __init__(self, h_dims=128, n_classes=10, n_branches=1, n_levels=1):
        super(ReadoutModule, self).__init__()

        self.predictor = nn.Linear(h_dims * 2, n_classes)
        self.n_branches = n_branches
        self.n_levels = n_levels

    def forward(self, t, lvl):
        #nodes = t[-self.n_branches ** self.n_levels:]
        nodes = t[:num_nodes(lvl, self.n_branches)]
        att = F.softmax(T.stack([node.att for node in nodes], 1), dim=1)
        h = T.stack([node.h for node in nodes], 1)
        return self.predictor((h * att).sum(dim=1))
        #h = T.stack([node.h for node in nodes], 1).mean(1)
        #return self.predictor(h)

parser = argparse.ArgumentParser(description='Alternative')
parser.add_argument('--row', default=200, type=int, help='image rows')
parser.add_argument('--col', default=200, type=int, help='image cols')
parser.add_argument('--n', default=5, type=int, help='number of epochs')
parser.add_argument('--log_interval', default=10, type=int, help='log interval')
parser.add_argument('--lr_where', default=2e-4, type=float, help='learning rate of where module')
parser.add_argument('--port', default=11111, type=int, help='visdom port')
parser.add_argument('--alter', action='store_true', help='indicates whether to use alternative training or not(joint training)')
parser.add_argument('--visdom', action='store_true', help='indicates whether to use visdom or not')
parser.add_argument('--share', action='store_true', help='indicates whether to share CNN params or not')
parser.add_argument('--fix', action='store_true', help='indicates whether to fix CNN or not')
parser.add_argument('--pretrain', action='store_true', help='pretrain or not pretrain')
parser.add_argument('--schedule', action='store_true', help='indicates whether to use schedule training or not')
parser.add_argument('--clip', default=0.1, type=float, help='gradient clipping norm')
parser.add_argument('--branches', default=2, type=int, help='branches')
parser.add_argument('--levels', default=2, type=int, help='levels')
args = parser.parse_args()
exp_setting = 'one_step_n_{}x{}_{}_{:.4f}_{}_{}_{}_{}'.format(args.row, args.col, args.n, args.lr_where,
                                                              'alter' if args.alter else 'joint',
                                                              'share' if args.share else 'noshare',
                                                              'fix' if args.fix else 'finetune',
                                                              'pretrain' if args.pretrain else 'fromscratch')

if not os.path.exists('logs/'):
    os.makedirs('logs/')
if not os.path.exists('logs/{}/'.format(exp_setting)):
    os.makedirs('logs/{}/'.format(exp_setting))
if args.schedule:
    for i in range(1, args.levels + 1):
        if not os.path.exists('logs/{}/{}'.format(exp_setting, i)):
            os.makedirs('logs/{}/{}'.format(exp_setting, i))

mnist_train = MNISTMulti('.', n_digits=1, backrand=0, image_rows=args.row, image_cols=args.col, download=True)
mnist_valid = MNISTMulti('.', n_digits=1, backrand=0, image_rows=args.row, image_cols=args.col, download=False, mode='valid')

n_branches = args.branches
n_levels = args.levels

builder = cuda(TreeBuilder(n_branches=n_branches, n_levels=n_levels))
readout = cuda(ReadoutModule(n_branches=n_branches, n_levels=n_levels))

train_shuffle = True

if args.visdom:
    wm = VisdomWindowManager(port=args.port)
n_epochs = args.n
len_train = len(mnist_train)
len_valid = len(mnist_valid)
phase = 'What' if args.alter else 'Joint'

loss_arr = []
acc_arr = []

if args.pretrain:
    pass
else:
    start_lvl = n_levels
    if args.schedule:
        start_lvl = 1
    for lvl in range(start_lvl, n_levels + 1):
        params = list(builder.parameters()) + list(readout.parameters())
        opt = T.optim.RMSprop(params, lr=1e-4)
        for epoch in range(n_epochs):
            print("Epoch {} starts...".format(epoch))

            batch_size = 64
            train_loader = data_generator(mnist_train, batch_size, train_shuffle)

            sum_loss = 0
            n_batches = len_train // batch_size
            hit = 0
            cnt = 0
            for i, (x, y, b) in enumerate(train_loader):
                t = builder(x, lvl)
                y_pred = readout(t, lvl)
                loss = F.cross_entropy(
                    y_pred, y
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.clip)
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
                    t = builder(x, lvl)
                    y_pred = readout(t, lvl)
                    loss = F.cross_entropy(
                        y_pred, y
                    )
                    sum_loss += loss.item()
                    hit += (y_pred.max(dim=-1)[1] == y).sum().item()
                    cnt += batch_size
                    if i == 0:
                        sample_imgs = x[:10]
                        length = num_nodes(lvl, n_branches)
                        sample_bboxs = [glimpse_to_xyhw(t[k].bbox[:10, :4] * 200) for k in range(1, length)]
                        sample_g = t[-1].g[:10]
                        statplot = StatPlot(5, 2)
                        statplot_g = StatPlot(5, 2)
                        for j in range(10):
                            statplot.add_image(
                                    sample_imgs[j][0],
                                    bboxs=[sample_bbox[j] for sample_bbox in sample_bboxs],
                                    clrs=['y', 'y', 'r', 'r', 'r', 'r'],
                                    )
                            statplot_g.add_image(sample_g[j][0])
                        if args.visdom:
                            wm.display_mpl_figure(statplot.fig, win='viz')
                            wm.display_mpl_figure(statplot_g.fig, win='vizg')
                            #wm.append_mpl_figure_to_sequence('bbox', statplot.fig)
                        else:
                            statplot.savefig('logs/{}/{}/epoch_{}.pdf'.format(exp_setting, lvl, epoch))

            avg_loss = sum_loss / i
            print("Loss on valid set: {}".format(avg_loss))
            print("Accuracy on valid set: {}".format(hit * 1.0 / cnt))
            if args.visdom:
                wm.append_scalar('loss', avg_loss)
                wm.append_scalar('acc', hit * 1.0 / cnt)
            else:
                loss_arr.append(avg_loss)
                acc_arr.append(hit * 1.0 / cnt)

            if hit * 1.0 / cnt > 0.9:
                if lvl < n_levels:
                    print("Accuracy achieve threshold, entering next level...")
                    break

        if not args.visdom:
            statplot = StatPlot(1, 2)
            statplot.add_curve(None, [loss_arr], labels=['loss'], title='loss curve', x_label='epoch', y_label='loss')
            statplot.add_curve(None, [acc_arr], labels=['acc'], title='acc curve', x_label='epoch', y_label='acc')
            statplot.savefig('logs/{}/{}/curve.pdf'.format(exp_setting), lvl)
