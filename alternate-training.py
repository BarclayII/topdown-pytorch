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

T.set_num_threads(4)

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
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--log_interval', default=10, type=int, help='log interval')
parser.add_argument('--share', action='store_true', help='indicates whether to share CNN params or not')
parser.add_argument('--pretrain', action='store_true', help='pretrain or not pretrain')
parser.add_argument('--schedule', action='store_true', help='indicates whether to use schedule training or not')
parser.add_argument('--att_type', default='naive', type=str, help='attention type: mean/naive/tanh')
parser.add_argument('--clip', default=0.1, type=float, help='gradient clipping norm')
parser.add_argument('--reg', default=1, type=float, help='regularization parameter')
parser.add_argument('--reg_type', default=0, type=int, help='regularization type')
parser.add_argument('--branches', default=2, type=int, help='branches')
parser.add_argument('--levels', default=2, type=int, help='levels')
parser.add_argument('--rank', action='store_true', help='use rank loss')
parser.add_argument('--backrand', default=0, type=int, help='background noise(randint between 0 to `backrand`)')
parser.add_argument('--glm_type', default='gaussian', type=str, help='glimpse type (gaussian, bilinear)')
parser.add_argument('--v_batch_size', default=256, type=int, help='valid batch size')
args = parser.parse_args()
expr_setting = '_'.join('{}-{}'.format(k, v) for k, v in vars(args).items() if k != 'resume' and k != 'v_batch_size')

writer = SummaryWriter('runs/{}'.format(expr_setting))
mnist_train = MNISTMulti('.', n_digits=1, backrand=args.backrand, image_rows=args.row, image_cols=args.col, download=True)
mnist_valid = MNISTMulti('.', n_digits=1, backrand=args.backrand, image_rows=args.row, image_cols=args.col, download=False, mode='valid')

n_branches = args.branches
n_levels = args.levels

if args.resume is not None:
    builder = T.load('checkpoints/builder_{}.pt'.format(args.resume))
    readout = T.load('checkpoints/readout_{}.pt'.format(args.resume))
else:
    builder = cuda(TreeBuilder(n_branches=n_branches,
                            n_levels=n_levels,
                            att_type=args.att_type,
                            c_reg=args.reg,
                            glimpse_type=args.glm_type,
                            reg_type=args.reg_type))
    readout = cuda(ReadoutModule(n_branches=n_branches, n_levels=n_levels))

train_shuffle = True

n_epochs = args.n
batch_size = args.batch_size

len_train = len(mnist_train)
len_valid = len(mnist_valid)

loss_arr = []
acc_arr = []

def rank_loss(a, b, margin=0):
    return (b - a + margin).clamp(min=0).mean()

def train():
    for epoch in range(n_epochs):
        print("Epoch {} starts...".format(epoch))
        params = list(builder.parameters()) + list(readout.parameters())
        opt = T.optim.RMSprop(params, lr=1e-4)

        start_lvl = 0 if args.schedule else n_levels
        train_loader = data_generator(mnist_train, batch_size, train_shuffle)
        sum_loss = 0
        n_batches = len_train // batch_size
        hit = 0
        cnt = 0
        levelwise_hit = np.zeros(n_levels + 1)

        for i, (x, y, b) in enumerate(train_loader):
            total_loss = 0

            t, loss_reg = builder(x)
            readout_list = readout(t)

            for lvl in range(start_lvl, n_levels + 1):
                y_pred, _ = readout_list[lvl]
                y_score = y_pred.gather(1, y[:, None])[:, 0]

                ce_loss = F.cross_entropy(y_pred, y)
                loss = ce_loss + loss_reg

                if args.rank and lvl > start_lvl:
                    loss_rank = rank_loss(y_score, y_score_last)
                    loss = loss + loss_rank
                y_score_last = y_score

                total_loss = total_loss + loss
                levelwise_hit[lvl] += (y_pred.max(dim=-1)[1] == y).sum().item()

            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            sum_loss += total_loss.item()
            hit = levelwise_hit[-1]
            cnt += batch_size

            if i % args.log_interval == 0 and i > 0:
                avg_loss = sum_loss / args.log_interval
                sum_loss = 0
                print('Batch {}/{}, loss = {}, acc = {}'.format(i, n_batches, avg_loss, hit * 1.0 / cnt))
                hit = 0
                cnt = 0
                levelwise_hit *= 0

        v_batch_size = args.v_batch_size
        valid_loader = data_generator(mnist_valid, v_batch_size, False)
        cnt = 0
        hit = 0
        levelwise_hit = np.zeros(n_levels + 1)
        sum_loss = 0
        with T.no_grad():
            for i, (x, y, b) in enumerate(valid_loader):
                total_loss = 0
                t, _ = builder(x)
                readout_list = readout(t)

                for lvl in range(start_lvl, n_levels + 1):
                    y_pred, att_weights = readout_list[lvl]
                    loss = F.cross_entropy(
                        y_pred, y
                    )
                    total_loss += loss
                    levelwise_hit[lvl] += (y_pred.max(dim=-1)[1] == y).sum().item()

                sum_loss += total_loss.item()
                hit = levelwise_hit[-1]
                cnt += v_batch_size

                lvl = n_levels

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
        levelwise_acc = levelwise_hit * 1.0 / cnt
        print("Loss on valid set: {}".format(avg_loss))
        print("Accuracy on valid set: {}".format(acc))

        writer.add_scalar('data/loss', avg_loss, epoch)
        if args.schedule:
            writer.add_scalars('data/levelwise_loss', {str(lvl): levelwise_acc[lvl] for lvl in range(n_levels + 1)}, epoch)
        writer.add_scalar('data/accuracy', acc, epoch)

        if (epoch + 1) % 10 == 0:
            print('Save checkpoint...')
            if not os.path.exists('checkpoints/'):
                os.makedirs('checkpoints')
            T.save(builder, 'checkpoints/builder_{}.pt'.format(epoch))
            T.save(readout, 'checkpoints/readout_{}.pt'.format(epoch))

if __name__ == '__main__':
    train()
