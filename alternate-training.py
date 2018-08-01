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
from viz import fig_to_ndarray_tb
from tensorboardX import SummaryWriter
from stats.utils import *
from modules import *
import tqdm

thres = [0.6, 0.95, 0.98, 0.98, 0.98]

def data_generator(dataset, batch_size, shuffle):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    for _x, _y, _B in dataloader:
        x = _x[:, None].expand(_x.shape[0], 3, _x.shape[1], _x.shape[2]).float() / 255.
        y = _y.squeeze(1)
        b = _B.squeeze(1).float() / 200
        yield cuda(x), cuda(y), cuda(b)

parser = argparse.ArgumentParser(description='Alternative')
parser.add_argument('--resume', default=None, help='resume training from checkpoint')
parser.add_argument('--row', default=200, type=int, help='image rows')
parser.add_argument('--col', default=200, type=int, help='image cols')
parser.add_argument('--n', default=100, type=int, help='number of epochs')
parser.add_argument('--log_interval', default=10, type=int, help='log interval')
parser.add_argument('--share', action='store_true', help='indicates whether to share CNN params or not')
parser.add_argument('--pretrain', action='store_true', help='pretrain or not pretrain')
parser.add_argument('--schedule', action='store_true', help='indicates whether to use schedule training or not')
parser.add_argument('--att_type', default='naive', type=str, help='attention type: mean/naive/self')
parser.add_argument('--clip', default=0.1, type=float, help='gradient clipping norm')
parser.add_argument('--reg', default=0, type=float, help='regularization parameter')
parser.add_argument('--branches', default=2, type=int, help='branches')
parser.add_argument('--levels', default=2, type=int, help='levels')
parser.add_argument('--backrand', default=0, type=int, help='background noise(randint between 0 to `backrand`)')
parser.add_argument('--glm_type', default='gaussian', type=str, help='glimpse type (gaussian, bilinear)')
args = parser.parse_args()
expr_setting = '_'.join('{}-{}'.format(k, v) for k, v in vars(args).items() if k is not 'resume')

writer = SummaryWriter('runs/{}'.format(expr_setting))
mnist_train = MNISTMulti('.', n_digits=1, backrand=0, image_rows=args.row, image_cols=args.col, download=True)
mnist_valid = MNISTMulti('.', n_digits=1, backrand=0, image_rows=args.row, image_cols=args.col, download=False, mode='valid')

n_branches = args.branches
n_levels = args.levels

if args.resume is not None:
    pass  # TODO

builder = cuda(TreeBuilder(n_branches=n_branches,
                           n_levels=n_levels,
                           att_type=args.att_type,
                           c_reg=args.reg,
                           glimpse_type=args.glm_type))
readout = cuda(ReadoutModule(n_branches=n_branches, n_levels=n_levels))

train_shuffle = True

n_epochs = args.n
len_train = len(mnist_train)
len_valid = len(mnist_valid)

loss_arr = []
acc_arr = []

start_lvl = n_levels
if args.schedule:
    start_lvl = 0
for lvl in range(start_lvl, n_levels + 1):
    print("Level {} start...".format(lvl))
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
            t, loss_reg = builder(x, lvl)
            y_pred, _ = readout(t, lvl)
            loss = F.cross_entropy(
                y_pred, y
            ) + loss_reg
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
                print('Batch {}/{}, loss = {}, acc={}'.format(i, n_batches, avg_loss, hit * 1.0 / cnt))
                hit = 0
                cnt = 0

        batch_size = 256
        valid_loader = data_generator(mnist_valid, batch_size, False)
        cnt = 0
        hit = 0
        sum_loss = 0
        with T.no_grad():
            for i, (x, y, b) in enumerate(valid_loader):
                t, _ = builder(x, lvl)
                y_pred, att_weights = readout(t, lvl)
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
                    sample_g_arr = [t[_].g[:10] for _ in range(length)]
                    statplot = StatPlot(5, 2)
                    statplot_g_arr = [StatPlot(5, 2) for _ in range(length)]
                    sample_atts = att_weights.cpu().numpy()[:10]
                    for j in range(10):
                        statplot.add_image(
                            sample_imgs[j][0],
                            bboxs=[sample_bbox[j] for sample_bbox in sample_bboxs],
                            clrs=['y', 'y', 'r', 'r', 'r', 'r'],
                            lws=sample_atts[j, 1:] * length
                        )
                        for k in range(length):
                            statplot_g_arr[k].add_image(sample_g_arr[k][j][0], title='att_weight={}'.format(sample_atts[j, k]))
                    writer.add_image('Image/{}/viz_bbox'.format(lvl), fig_to_ndarray_tb(statplot.fig), epoch)
                    for k in range(length):
                        writer.add_image('Image/{}/viz_glim_{}'.format(lvl, k), fig_to_ndarray_tb(statplot_g_arr[k].fig), epoch)
                    plt.close('all')

        avg_loss = sum_loss / i
        acc = hit * 1.0 / cnt
        print("Loss on valid set: {}".format(avg_loss))
        print("Accuracy on valid set: {}".format(acc))

        writer.add_scalar('data/{}/loss'.format(lvl), avg_loss, epoch)
        writer.add_scalar('data/{}/accuracy'.format(lvl), acc, epoch)

        if (epoch + 1) % 10 == 0:
            print('Save checkpoint...')
            if not os.path.exists('checkpoints/'):
                os.makedirs('checkpoints')
            T.save(builder, 'checkpoints/builder_{}_{}.pt'.format(lvl, epoch))
            T.save(readout, 'checkpoints/readout_{}_{}.pt'.format(lvl, epoch))

        if acc > thres[lvl]:
            if lvl < n_levels:
                print("Accuracy achieve threshold, entering next level...")
                break
