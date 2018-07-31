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
from modules import *
import tqdm

def data_generator(dataset, batch_size, shuffle):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    for _x, _y, _B in dataloader:
        x = _x[:, None].expand(_x.shape[0], 3, _x.shape[1], _x.shape[2]).float() / 255.
        y = _y.squeeze(1)
        b = _B.squeeze(1).float() / 200
        yield cuda(x), cuda(y), cuda(b)

parser = argparse.ArgumentParser(description='Alternative')
parser.add_argument('--row', default=200, type=int, help='image rows')
parser.add_argument('--col', default=200, type=int, help='image cols')
parser.add_argument('--n', default=5, type=int, help='number of epochs')
parser.add_argument('--log_interval', default=10, type=int, help='log interval')
parser.add_argument('--lr_where', default=2e-4, type=float, help='learning rate of where module')
parser.add_argument('--decay', action='store_true', help='indicates whether to deacy lr where or not')
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
exp_setting = 'one_step_n_{}x{}_{}_{:.4f}_{}_{}_{}_{}{}'.format(args.row, args.col, args.n, args.lr_where,
                                                              'alter' if args.alter else 'joint',
                                                              'share' if args.share else 'noshare',
                                                              'fix' if args.fix else 'finetune',
                                                                'pretrain' if args.pretrain else 'fromscratch',
                                                                '-decay' if args.decay else '')

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
    #start_lvl = n_levels
    #if args.schedule:
    #    start_lvl = 1
    for epoch in range(n_epochs):
        params = list(builder.parameters()) + list(readout.parameters())
        opt = T.optim.RMSprop(params, lr=1e-4)
        start_lvl = 1 if args.schedule else n_levels
        for lvl in range(start_lvl, n_levels + 1):
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
