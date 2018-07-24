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
            nn.Linear(h_dims, n_classes),
        )

    def forward(self, glimpse_kxk):
        batch_size = glimpse_kxk.shape[0]
        return self.net_h(self.cnn(glimpse_kxk).view(batch_size, -1))

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
            nn.Linear(h_dims, g_dims),
        )

    def forward(self, glimpse_kxk):
        batch_size = glimpse_kxk.shape[0]
        return self.net_g(self.cnn(glimpse_kxk).view(batch_size, -1))

glimpse = create_glimpse('gaussian', (15, 15))

filters = [16, 32, 64, 128, 256]
kernel_size = (3, 3)
final_pool_size = (2, 2)
h_dims = 128
n_classes = 10
g_dims = glimpse.att_params

net_phi = WhatModule(filters, kernel_size, final_pool_size, h_dims, n_classes)
net_g = WhereModule(filters, kernel_size, final_pool_size, h_dims, g_dims)
net_phi = cuda(net_phi)
net_g = cuda(net_g)

parser = argparse.ArgumentParser(description='Alternative')
parser.add_argument('--pretrain', action='store_true', help='pretrain or not pretrain')
parser.add_argument('--row', default=100, help='image rows')
parser.add_argument('--col', default=100, help='image cols')
parser.add_argument('--n', default=5, help='number of epochs')
parser.add_argument('--log_interval', default=10)
args = parser.parse_args()

mnist_train = MNISTMulti('.', n_digits=1, backrand=0, image_rows=args.row, image_cols=args.col, download=True)
mnist_valid = MNISTMulti('.', n_digits=1, backrand=0, image_rows=args.row, image_cols=args.col, download=False)

batch_size = 64
train_shuffle = True

if args.pretrain:
    pass
else:
    n_epochs = args.n
    len_train = len(mnist_train)
    len_valid = len(mnist_valid)
    phase = 'What'
    for epoch in range(2 * n_epochs):
        print("Epoch {} starts...".format(epoch))

        train_loader = data_generator(mnist_train, batch_size, train_shuffle)
        if phase == 'What':
            opt = optim.RMSprop(
                [
                    {'params': net_phi.parameters(), 'lr': 1e-3},
                    {'params': net_g.parameters(), 'lr': 0}
                ]
            )
            phase = 'Where'
        else:
            opt = optim.RMSprop(
                [
                    {'params': net_phi.parameters(), 'lr': 0},
                    {'params': net_g.parameters(), 'lr': 1e-5}
                ]
            )
            phase = 'What'

        sum_loss = 0
        n_batches = len_train // batch_size
        hit = 0
        cnt = 0
        for i, (x, y, b) in enumerate(train_loader):
            g, _ = glimpse.rescale(
                cuda(b.new(batch_size, g_dims).zero_())[:, None], False)
            x_glim = glimpse(x, g)[:, 0]
            g_pred = net_g(x_glim)
            g, _ = glimpse.rescale(
                g_pred[:, None], False)
            x_glim = glimpse(x, g)[:, 0]
            y_pred = net_phi(x_glim)
            hit += (y_pred.max(dim=-1)[1] == y).sum().item()
            cnt += batch_size
            loss = F.cross_entropy(
                y_pred, y
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_loss += loss.item()
            if i % args.log_interval == 0 and i > 0:
                avg_loss = sum_loss / args.log_interval
                sum_loss = 0
                print('{} phase, Batch {}/{}, loss = {}, acc={}'.format(phase, i, n_batches, avg_loss, hit * 1.0 / cnt))
                hit = 0
                cnt = 0

        T.save(net_phi, 'epoch_{}_what.pt'.format(n_epochs))
        T.save(net_g, 'epoch_{}_where.pt'.format(n_epochs))
