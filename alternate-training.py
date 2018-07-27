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

if args.pretrain:
    with open('what_net.pt', 'rb') as f:
        net_phi = T.load(f)
else:
    net_phi = WhatModule(filters, kernel_size, final_pool_size, h_dims, n_classes)
cnn = None
if args.share:
    cnn = net_phi.cnn
else:
    filters = [16, 32]

net_g = WhereModule(filters, kernel_size, final_pool_size, h_dims, g_dims, cnn=cnn)
net_phi = cuda(net_phi)
net_g = cuda(net_g)

mnist_train = MNISTMulti('.', n_digits=1, backrand=0, image_rows=args.row, image_cols=args.col, download=True)
mnist_valid = MNISTMulti('.', n_digits=1, backrand=0, image_rows=args.row, image_cols=args.col, download=False, mode='valid')

train_shuffle = True

if args.visdom:
    wm = VisdomWindowManager(port=args.port)
n_epochs = args.n
len_train = len(mnist_train)
len_valid = len(mnist_valid)
phase = 'What' if args.alter else 'Joint'
lr_phi = 1e-3
if args.fix:
    lr_phi = 0
lr_g = args.lr_where

loss_arr = []
acc_arr = []
for epoch in range(n_epochs):
    print("Epoch {} starts...".format(epoch))

    batch_size = 64
    train_loader = data_generator(mnist_train, batch_size, train_shuffle)

    if args.share:
        net_g_params = net_g.net_g.parameters()
    else:
        net_g_params = net_g.parameters()

    if phase == 'What':
        opt = optim.RMSprop(
            [
                {'params': net_phi.parameters(), 'lr': lr_phi},
                {'params': net_g_params, 'lr': 0}
            ]
        )
    elif phase == 'Where':
        opt = optim.RMSprop(
            [
                {'params': net_phi.parameters(), 'lr': 0},
                {'params': net_g_params, 'lr': lr_g}
            ]
        )
    elif phase == "Joint":
        opt = optim.RMSprop(
            [
                {'params': net_phi.parameters(), 'lr': lr_phi},
                {'params': net_g_params, 'lr': lr_g}
            ]
        )
    else:
        assert False

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

    phase = 'Where' if phase is 'What' else ('What' if phase is 'Where' else phase)

    batch_size = 256
    valid_loader = data_generator(mnist_valid, batch_size, False)
    cnt = 0
    hit = 0
    sum_loss = 0
    with T.no_grad():
        for i, (x, y, b) in enumerate(valid_loader):
            g, _ = glimpse.rescale(
                cuda(b.new(batch_size, g_dims).zero_())[:, None], False)
            x_glim = glimpse(x, g)[:, 0]
            g_pred = net_g(x_glim)
            g, _ = glimpse.rescale(
                g_pred[:, None], False)
            x_glim = glimpse(x, g)[:, 0]
            y_pred = net_phi(x_glim)
            loss = F.cross_entropy(
                y_pred, y
            )
            sum_loss += loss.item()
            hit += (y_pred.max(dim=-1)[1] == y).sum().item()
            cnt += batch_size
            if i == 0:
                sample_imgs = x[:10]
                sample_gs = g[:10, 0, :]
                sample_bboxs = glimpse_to_xyhw(sample_gs) * 200
                sample_glims = x_glim[:10]
                sample_probs = F.softmax(y_pred[:10], dim=-1)
                statplot = StatPlot(10, 3)
                for j in range(10):
                    statplot.add_image(sample_imgs[j][0], bboxs=[sample_bboxs[j]], title='raw image and bbox')
                    statplot.add_image(sample_glims[j][0], title='glimpse image')
                    statplot.add_bar(None, [sample_probs[j]], labels=['prob'], title='prediction')

                if args.visdom:
                    wm.display_mpl_figure(statplot.fig)
                    wm.append_mpl_figure_to_sequence('bbox', statplot.fig)
                else:
                    statplot.savefig('logs/{}/epoch_{}.pdf'.format(exp_setting, epoch))

        avg_loss = sum_loss / i
    print("Loss on valid set: {}".format(avg_loss))
    print("Accuracy on valid set: {}".format(hit * 1.0 / cnt))
    if args.visdom:
        wm.append_scalar('loss', avg_loss)
        wm.append_scalar('acc', hit * 1.0 / cnt)
    else:
        loss_arr.append(avg_loss)
        acc_arr.append(hit * 1.0 / cnt)

if not args.visdom:
    statplot = StatPlot(1, 2)
    statplot.add_curve(None, [loss_arr], labels=['loss'], title='loss curve', x_label='epoch', y_label='loss')
    statplot.add_curve(None, [acc_arr], labels=['acc'], title='acc curve', x_label='epoch', y_label='acc')
    statplot.savefig('logs/{}/curve.pdf'.format(exp_setting))
