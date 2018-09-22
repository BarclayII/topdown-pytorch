"""
TODO's:
1. Scheduler
2. Accelerate Nearest Neighbors (Currently we make it an option in argparse)
3. Visualization of CNN feature maps
4. Levelwise early stop
5. Failure cases
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torchvision.models import resnet18
import numpy as np
import os
import sys
from glimpse import create_glimpse
import util
from util import cuda
from datasets import get_generator
from viz import *
from tensorboardX import SummaryWriter
from stats import *
from modules import *
from configs import *
import tqdm

T.set_num_threads(4)

train_loader, valid_loader, test_loader, preprocessor = get_generator(args)

writer = SummaryWriter('runs/{}'.format(expr_setting))

n_branches = args.branches
n_levels = args.levels

if args.resume is not None:
    builder = T.load('checkpoints/{}_builder_{}.pt'.format(expr_setting, args.resume))
    readout = T.load('checkpoints/{}_readout_{}.pt'.format(expr_setting, args.resume))
else:
    regularizer_classes = {
        PCRegularizer: args.pc_coef,
        CCRegularizer: args.cc_coef,
        ResRegularizer: args.res_coef
    }
    network_params = NETWORK_PARAMS[args.dataset]
    builder = cuda(nn.DataParallel(TreeBuilder(n_branches=n_branches,
                            n_levels=n_levels,
                            n_classes=network_params['n_classes'],
                            regularizer_classes=regularizer_classes,
                            glimpse_type=args.glm_type,
                            glimpse_size=(args.glm_size, args.glm_size),
                            fm_target_size=network_params['fm_target_size'],
                            final_pool_size=network_params['final_pool_size'],
                            final_n_channels=network_params['final_n_channels'],
                            what__cnn=network_params['cnn'],
                            what__fix=args.fix,
                            what__in_dims=network_params['in_dims'])))
    readout = cuda(nn.DataParallel(
        create_readout('alpha',
                       final_n_channels=network_params['final_n_channels'],
                       n_branches=n_branches,
                       n_levels=n_levels,
                       n_classes=network_params['n_classes'])))

train_shuffle = True

n_epochs = args.n
batch_size = args.batch_size

loss_arr = []
acc_arr = []

logfile = open('debug.log', 'w')
normalize = getattr(util, network_params['normalize'])
normalize_inverse = getattr(util, network_params['normalize_inverse'])

#@profile
def train():
    lvl_turn = 0
    coef_lvl = [1 for _ in range(n_levels + 1)]
    best_epoch = [0 for _ in range(n_levels + 1)]
    best_valid_loss = [1e6 for _ in range(n_levels + 1)]
    levels = args.levels_from
    n_train_batches = len(train_loader)

    params = list(builder.parameters()) + list(readout.parameters())
    """
    if args.dataset.startswith('mnist'):
        lr = 1e-4
        opt = T.optim.RMSprop(params, lr=1e-4)
    elif args.dataset in dataset_with_sgd_schedule:
        lr = 0.01
        opt = T.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-5)
        #opt = T.optim.RMSprop(params, lr=1e-4, weight_decay=5e-4)
    elif args.dataset in ['imagenet', 'dogs']:
        lr = 0.1
        opt = T.optim.RMSprop(params, lr=3e-5, weight_decay=1e-4)
    elif args.dataset == 'flower' or args.dataset == 'bird':
        lr = 1e-4
        #opt = T.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-5)
        opt = T.optim.RMSprop(params, lr=1e-4)
    """

    opt_params = OPTIM_PARAMS[args.dataset]
    opt = create_optim(
        opt_params['mode'],
        params,
        opt_params['args']
    )

    scheduler = create_scheduler(
        opt_params['scheduler_mode'],
        opt,
        opt_params['scheduler_args']
    )
    print('initial lr is {}'.format(opt.param_groups[0]['lr']))

    for epoch in range(n_epochs):
        print("Epoch {} starts...".format(epoch))

        readout_start_lvl = 0 #levels
        sum_loss = 0
        train_loss_dict = {
            'pc': 0.,
            'cc': 0.,
            'ce': 0,
            'res': 0,
        }
        hit = 0
        cnt = 0
        levelwise_hit = np.zeros(n_levels + 1)
        levelwise_loss = np.zeros(n_levels + 1)

        builder.train(mode=True)
        readout.train(mode=True)
        # I can directly use tqdm.tqdm(train_loader) but it doesn't display progress bar and time estimates
        with tqdm.tqdm(train_loader) as tq:
            #x, y, b = preprocessor(next(train_iter))
            for i, item in enumerate(tq):
                x, y, b = preprocessor(item)
                print(x.shape, y.shape, file=logfile)

                x_in = normalize(x)

                total_loss = 0

                t, (loss_pc, loss_cc, loss_res) = builder(x_in, levels)
                loss_pc = loss_pc.mean()
                loss_cc = loss_cc.mean()
                loss_res = loss_res.mean()
                train_loss_dict['pc'] += loss_pc.item()
                train_loss_dict['cc'] += loss_cc.item()
                train_loss_dict['res'] += loss_res.item()
                readout_list = readout(t, levels)

                total_loss = loss_pc + loss_cc + loss_res
                for lvl in range(readout_start_lvl, levels + 1):
                    y_pred  = readout_list[lvl]
                    y_score = y_pred.gather(1, y[:, None])[:, 0]
                    loss_ce = F.cross_entropy(y_pred, y) * coef_lvl[lvl]
                    train_loss_dict['ce'] += loss_ce.item()
                    levelwise_loss[lvl] += loss_ce.item()
                    loss = loss_ce

                    y_score_last = y_score

                    total_loss = total_loss + loss
                    current_hit = (y_pred.max(dim=-1)[1] == y).sum().item()
                    levelwise_hit[lvl] += current_hit

                opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(params, 0.1)
                opt.step()
                sum_loss += total_loss.item()
                hit = levelwise_hit[levels] # - 1]
                cnt += batch_size

                if i == 0:
                    sample_imgs = x[:10]
                    bbox_scaler = T.FloatTensor([[x.shape[3], x.shape[2], x.shape[3], x.shape[2]]]).to(x)
                    length = len(t)
                    sample_bboxs = [
                            glimpse_to_xyhw(t[k].b[:10, :4].detach()) * bbox_scaler
                            for k in range(1, length)
                            ]
                    sample_g_arr = [
                        normalize_inverse(t[_].g[:10].detach()) for _ in range(length)]
                    viz(epoch, sample_imgs, sample_bboxs, sample_g_arr, 'train', writer,
                            n_branches=n_branches, n_levels=n_levels)

                tq.set_postfix({
                    'train_loss': total_loss.item(),
                    'train_acc': current_hit / batch_size,
                    })
                if i == n_train_batches - 1:
                    tq.set_postfix({
                        'train_avg_loss': sum_loss / n_train_batches,
                        'train_avg_acc': hit / cnt,
                        })
        for key in train_loss_dict.keys():
            train_loss_dict[key] /= n_train_batches
        writer.add_scalars('data/train_loss_dict', train_loss_dict, epoch)
        levelwise_loss = levelwise_loss * 1.0 / i
        levelwise_acc = levelwise_hit * 1.0 / cnt
        writer.add_scalars('data/train_levelwise_acc', {str(lvl): levelwise_acc[lvl] for lvl in range(levels + 1)}, epoch)
        writer.add_scalars('data/train_levelwise_loss', {str(lvl): levelwise_loss[lvl] for lvl in range(levels + 1)}, epoch)

        # Validation phase
        hit = 0
        cnt = 0
        levelwise_hit = np.zeros(n_levels + 1)
        levelwise_loss = np.zeros(n_levels + 1)
        sum_loss = 0
        builder.eval()
        readout.eval()
        with T.no_grad():
            for i, item in enumerate(tqdm.tqdm(valid_loader)):
                x, y, b = preprocessor(item)
                print(x.shape, y.shape, file=logfile)
                if args.dataset in dataset_with_normalize:
                    x_in = imagenet_normalize(x)
                else:
                    x_in = x

                total_loss = 0
                t, _ = builder(x_in, levels)
                readout_list = readout(t, levels)

                for lvl in range(readout_start_lvl, levels + 1):
                    y_pred = readout_list[lvl]

                    loss = F.cross_entropy(
                        y_pred, y
                    ) * coef_lvl[lvl]
                    total_loss += loss
                    levelwise_loss[lvl] += loss.item()
                    levelwise_hit[lvl] += (y_pred.max(dim=-1)[1] == y).sum().item()

                sum_loss += total_loss.item()
                hit = levelwise_hit[levels]
                cnt += args.v_batch_size

                if i == 0:
                    sample_imgs = x[:10]
                    length = len(t)
                    bbox_scaler = T.FloatTensor([[x.shape[3], x.shape[2], x.shape[3], x.shape[2]]]).to(x)
                    sample_bboxs = [
                            glimpse_to_xyhw(t[k].b[:10, :4].detach()) * bbox_scaler
                            for k in range(1, length)
                            ]
                    sample_g_arr = [
                        normalize_inverse(t[_].g[:10]) for _ in range(length)]

                    viz(epoch, sample_imgs, sample_bboxs, sample_g_arr, 'valid', writer,
                            n_branches=n_branches, n_levels=levels)

                    if args.nearest:
                        # nearest neighbor construction
                        nnset = NearestNeighborImageSet(
                                sample_imgs.permute(0, 2, 3, 1),
                                T.cat(readout.module.hs, 1)[:10],
                                bboxs=sample_bboxs,
                                clrs=getclrs(n_branches, levels),
                                title=[str(_y_pred) + '/' + str(_y)
                                    for _y_pred, _y in zip(y_pred.max(-1)[1], y)],
                                )

                        print()
                        for j, item_train in tqdm.tqdm(enumerate(train_loader)):
                            x, y, b = preprocessor(item_train)
                            print(x.shape, y.shape, file=logfile)
                            x_in = normalize(x)

                            t, _ = builder(x_in, levels)
                            readout_list = readout(t, levels)
                            bbox_scaler = T.FloatTensor([[x.shape[3], x.shape[2], x.shape[3], x.shape[2]]]).to(x)
                            sample_bboxs = [
                                    glimpse_to_xyhw(t[k].b[:, :4].detach()) * bbox_scaler
                                    for k in range(1, length)
                            ]
                            nnset.push(
                                    x.permute(0, 2, 3, 1),
                                    T.cat(readout.module.hs, 1),
                                    bboxs=sample_bboxs,
                                    clrs=getclrs(n_branches, levels),
                                    title=[str(_y) for _y in y],
                            )

                        nnset.display()
                        writer.add_image('Image/val/nn', fig_to_ndarray_tb(nnset.stat_plot.fig), epoch)

        avg_loss = sum_loss / i
        levelwise_loss = levelwise_loss * 1.0 / i
        acc = hit * 1.0 / cnt
        levelwise_acc = levelwise_hit * 1.0 / cnt
        print("Loss on valid set: {}".format(avg_loss))
        print("Accuracy on valid set: {}".format(acc))

        writer.add_scalar('data/loss', avg_loss, epoch)
        if args.schedule:
            writer.add_scalars('data/levelwise_acc', {str(lvl): levelwise_acc[lvl] for lvl in range(levels + 1)}, epoch)
            writer.add_scalars('data/levelwise_loss', {str(lvl): levelwise_loss[lvl] for lvl in range(levels + 1)}, epoch)
        writer.add_scalar('data/accuracy', acc, epoch)

        if (epoch + 1) % 10 == 0:
            print('Save checkpoint...')
            if not os.path.exists('checkpoints/'):
                os.makedirs('checkpoints')
            T.save(builder, 'checkpoints/{}_builder_{}.pt'.format(expr_setting, epoch))
            T.save(readout, 'checkpoints/{}_readout_{}.pt'.format(expr_setting, epoch))

        if best_valid_loss[lvl_turn] > levelwise_loss[lvl_turn]:
            best_valid_loss[lvl_turn] = levelwise_loss[lvl_turn]
            T.save(builder.state_dict(), 'checkpoints/{}_builder_best.pt'.format(expr_setting))
            T.save(readout.state_dict(), 'checkpoints/{}_readout_best.pt'.format(expr_setting))
            best_epoch[lvl_turn] = epoch
        elif (best_epoch[lvl_turn] <= epoch - 5 or epoch == n_epochs - 1) and test_loader is not None:
            print('Early Stopping on level {}...'.format(lvl_turn))
            coef_lvl[lvl_turn] = 0
            lvl_turn += 1
            builder.load_state_dict(T.load('checkpoints/{}_builder_best.pt'.format(expr_setting)))
            readout.load_state_dict(T.load('checkpoints/{}_readout_best.pt'.format(expr_setting)))
            if lvl_turn == n_levels + 1:
                cnt = 0
                hit = 0
                levelwise_hit = np.zeros(n_levels + 1)
                sum_loss = 0
                with T.no_grad():
                    for i, item in enumerate(tqdm.tqdm(test_loader)):
                        x, y, b = preprocessor(item)
                        x_in = normalize(x)

                        total_loss = 0
                        t, _ = builder(x_in, levels)
                        readout_list = readout(t, levels)

                        for lvl in range(readout_start_lvl, levels + 1):
                            y_pred = readout_list[lvl]
                            loss = F.cross_entropy(
                                y_pred, y
                            ) * coef_lvl[lvl]
                            total_loss += loss
                            levelwise_hit[lvl] += (y_pred.max(dim=-1)[1] == y).sum().item()

                        sum_loss += total_loss.item()
                        hit = levelwise_hit[levels]
                        cnt += args.v_batch_size

                avg_loss = sum_loss / i
                acc = hit * 1.0 / cnt
                levelwise_acc = levelwise_hit * 1.0 / cnt
                print("Loss on test set: {}".format(avg_loss))
                print("Accuracy on test set: {}".format(acc))

                for lvl in range(levels + 1):
                    print("Levelwise accuracy on level {}: {}".format(lvl, levelwise_acc[lvl]))
                break
        print('learning rate is {}'.format(opt.param_groups[0]['lr']))
        scheduler.step()

if __name__ == '__main__':
    train()
