import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from glimpse import create_glimpse
from util import cuda
import matplotlib.pyplot as plt

def visualize(fig):
    import matplotlib.pyplot as plt
    plt.imshow(fig.numpy(), cmap='gray')
    plt.show()

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

filters = [16, 32, 64, 128, 256]
kernel_size = (3, 3)
final_pool_size = (2, 2)
h_dims = 128
n_classes = 10
batch_size = 32

glimpse = create_glimpse('gaussian', (15, 15))
g_dims = glimpse.att_params

import os
from datasets import MNISTMulti

mnist_train = MNISTMulti('.', n_digits=1, backrand=0, image_rows=200, image_cols=200,
                         download=True)
mnist_valid = MNISTMulti('.', n_digits=1, backrand=0, image_rows=200, image_cols=200,
                         download=False, mode='valid')
train_shuffle = True
valid_shuffle = False

def bbox_to_glimpse(b):
    c_x = b[:, 0]
    c_y = b[:, 1]
    d_x = b[:, 2]
    d_y = b[:, 3]
    s_x = d_x / 2
    s_y = d_y / 2
    return T.stack([c_x, c_y, d_x, d_y, s_x, s_y], dim=-1)

def data_generator(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    for _x, _y, _B in dataloader:
        x = _x[:, None].expand(_x.shape[0], 3, _x.shape[1], _x.shape[2]).float() / 255.
        y = _y.squeeze(1)
        b = _B.squeeze(1).float() / 200
        yield cuda(x), cuda(y), cuda(b)

def init_canvas():
    fig, ax = plt.subplots(4, 2)
    fig.set_size_inches(8, 16)
    return fig, ax

def display_curve(fig, ax, i, j, val, title):
    y = np.maximum(np.minimum(val.numpy(), 50), -50)
    x = np.arange(len(y))
    ax[i, j].plot(x, y, color='b')
    ax[i, j].plot(x, [0] * len(y), color='g')
    ax[i, j].set_title(title)

def display_image(fig, ax, i, j, image, title):
    ax[i, j].imshow(image, cmap='gray')
    ax[i, j].set_title(title)

if os.path.exists('grad_cnn.pt'):
    cnn = T.load('grad_cnn.pt')
    net_h = T.load('grad_net_h.pt')
    #plt.subplots_adjust(wspace=0, hspace =100)
    valid_loader = data_generator(mnist_valid, 1, valid_shuffle)
    whole_glim = cuda(T.tensor([[0.5, 0.5, 1.0, 1.0, 0.5, 0.5]]))
    cnt = 0
    for x, y, b in valid_loader:
        glim = bbox_to_glimpse(b)
        grads = []
        losses = []
        for j in np.linspace(0, 1, 41):
            new_glim = glim * j + whole_glim * (1 - j)
            new_glim.requires_grad = True
            g = glimpse(x, new_glim.unsqueeze(1))[:, 0]
            if j == 0.:
                g_first = g[0][0].detach().cpu()
            if j == 1.:
                g_last = g[0][0].detach().cpu()
            out = net_h(cnn(g).view(1, -1))
            loss = F.cross_entropy(
                out, y
            )
            loss.backward()
            grads.append(new_glim.grad.cpu())
            losses.append(loss.item())
            cnn.zero_grad()
            net_h.zero_grad()
        grad_mat = T.cat(grads, dim=0)
        loss_mat = T.tensor(losses)
        fig, ax = init_canvas()
        display_curve(fig, ax, 0, 0, grad_mat[:, 0], 'gradient of cx')
        display_curve(fig, ax, 0, 1, grad_mat[:, 1], 'gradient of cy')
        display_curve(fig, ax, 1, 0, grad_mat[:, 2], 'gradient of dx')
        display_curve(fig, ax, 1, 1, grad_mat[:, 3], 'gradient of dy')
        display_curve(fig, ax, 2, 0, grad_mat[:, 4], 'gradient of sx')
        display_curve(fig, ax, 2, 1, grad_mat[:, 5], 'gradient of sy')
        display_curve(fig, ax, 3, 0, loss_mat, 'loss')
        display_image(fig, ax, 3, 1, T.cat([g_first, g_last], dim=-1), 'image first/last')
        fig.savefig('{}.pdf'.format(cnt))
        cnt += 1
        if cnt == 20:
            break

else:

    cnn = build_cnn(
        filters=filters,
        kernel_size=kernel_size,
        final_pool_size=final_pool_size
    )
    net_h = nn.Sequential(
        nn.Linear(filters[-1] * np.prod(final_pool_size), h_dims),
        nn.ReLU(),
        nn.Linear(h_dims, n_classes),
    )
    cnn = cuda(cnn)
    net_h = cuda(net_h)

    len_train = len(mnist_train)
    len_valid = len(mnist_valid)
    n_epochs = 2
    log_interval = 10

    opt = optim.Adam(
            [
                {"params": cnn.parameters()},
                {"params": net_h.parameters()}
            ], lr=1e-3)

    for epoch in range(n_epochs):
        train_loader = data_generator(mnist_train, batch_size, train_shuffle)
        valid_loader = data_generator(mnist_valid, batch_size, valid_shuffle)
        n_batches = len_train // batch_size
        sum_loss = 0
        for i, (x, y, b) in enumerate(train_loader):
            glim = bbox_to_glimpse(b)
            g = glimpse(x, glim[:, None])[:, 0]

            out = net_h(cnn(g).view(batch_size, -1))
            loss = F.cross_entropy(
                out, y
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_loss += loss.item()
            if i % log_interval == 0 and i > 0:
                avg_loss = sum_loss / log_interval
                print("Batch {}/{}, loss = {}".format(i, n_batches, avg_loss))
                sum_loss = 0

        tot = 0
        hit = 0
        for x, y, b in valid_loader:
            glim = bbox_to_glimpse(b)
            g = glimpse(x, glim[:, None])[:, 0]
            out = net_h(cnn(g).view(batch_size, -1))
            pred = out.max(dim=-1)[1]
            hit += (pred == y).sum().item()
            tot += y.shape[0]
        print(hit, tot)
        print("Epoch {}/{}, accuracy = {}".format(epoch, n_epochs, hit / tot))

    T.save(cnn, 'grad_cnn.pt')
    T.save(net_h, 'grad_net_h.pt')
