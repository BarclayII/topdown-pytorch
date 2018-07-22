"""
Description: https://hackmd.io/ROptX8dnQRyIGhVQL_8k2A
For convenience, here we use `h` to denote `\phi`.

Message Module:
    s_t = GRU(a_{t-1}, a{t-2}, ..., a{0})

Update Module: (x is given)
    b_t = f_glim(s_t)
    h_t = f_h(x, b_t)
    a_t = (h_t, b_t)
    c_t = f_c(a_t)

Readout Module:
    Trajectory: s_0, a_0, s_1, a_1, ... s_T, a_T
    //c_i = c_i / (\sum{c_i})
    //y_T = f_out(\sum_{i=1}^{T} c_i h_i)
    y_i = f_out(h_i)

Node states:
    s: vec(h_dims)
    a: tuple(h, b)
    g: glimpse image
    //c: attention weight

Networks:
    to_state: GRU # h_dims + g_dims -> h_dims
    f_glim:   MLP # h_dims -> g_dims(6)
    f_h:      MLP(CNN) # glimpse_size(15 x 15) -> h_dims
    //f_c:      MLP # h_dims + g_dims   -> 1
    f_out:    MLP # h_dims -> n_classes

Variations:
    - Train from scratch / Fine-tuned / Pre-trained
    - Share weights / Not share weights (within the same layer)
"""

import networkx as nx
from glimpse import create_glimpse
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as MODELS
import torch.nn.init as INIT
from util import USE_CUDA, cuda
import numpy as np
import skorch
from viz import VisdomWindowManager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dgl.graph import DGLGraph

batch_size = 32
wm = VisdomWindowManager(port=11111)

def visualize(fig):
    plt.imshow(fig, cmap='gray')
    plt.show()

def build_cnn(**config):
    cnn_layer_list = []
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
        cnn_layer_list.append(module)

        if i < len(filters) - 1:
            cnn_layer_list.append(nn.LeakyReLU())
        cnn_layer_list.append(nn.AdaptiveMaxPool2d(final_pool_size))

    return nn.Sequential(*cnn_layer_list)

class MessageModule(nn.Module):
    def __init__(self, **kwargs):
        super(MessageModule, self).__init__()
        h_dims = kwargs['h_dims']
        g_dims = kwargs['g_dims']
        self.to_state = nn.GRUCell(h_dims + g_dims, h_dims)
        INIT.orthogonal_(self.to_state.weight_hh)

    def forward(self, src, dst, edge):
        return self.to_state(T.cat(src['a'], dim=-1), src['s'])

class UpdateModule(nn.Module):
    """
    UpdateModule:
    Input:
        s_t: (a_{t-1})

    Returns:
        a_t: (h_t, b_t)
        c_t: attention weight
    """

    def __init__(self, **config):
        """
        h_dims=128,
        n_classes=10,
        filters=[16, 32, 64, 128, 256],
        kernel_size=(3, 3),
        final_pool_size=(2, 2),
        glimpse_type='gaussian',
        glimpse_size=(15, 15),
        cnn='cnn'
        """
        super(UpdateModule, self).__init__()
        glimpse_type = config['glimpse_type']
        glimpse_size = config['glimpse_size']
        self.glimpse = create_glimpse(glimpse_type, glimpse_size)

        h_dims = config['h_dims']
        n_classes = config['n_classes']
        g_dims = self.glimpse.att_params

        self.f_glim = nn.Sequential(
            nn.Linear(h_dims, h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, g_dims),
        )

        self.f_c = nn.Sequential(
            nn.Linear(h_dims + g_dims, h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, 1),
            nn.Sigmoid(),
        )

        cnn = config['cnn']
        final_pool_size = config['final_pool_size']

        if True: # Not pretrained
            filters = config['filters']
            kernel_size = config['kernel_size']
            self.cnn = build_cnn(
                filters=filters,
                kernel_size=kernel_size,
                final_pool_size=final_pool_size,
            )
            self.f_h = nn.Linear(filters[-1] * np.prod(final_pool_size), h_dims)

        self.h_dims = h_dims

    def set_image(self, x):
        self.x = x

    def forward(self, node_state, message):
        batch_size = node_state['s'].shape[0]

        if len(message) == 0: # root node
            s = node_state['s'].new(batch_size, self.h_dims).zero_()
        else:
            s = T.stack(message).mean(0)

        b_new = self.f_glim(s)
        b_rescaled, _ = self.glimpse.rescale(b_new[:, None], False)
        g = self.glimpse(self.x, b_rescaled)[:, 0]
        h_new = self.f_h(self.cnn(g).view(batch_size, -1))
        c = self.f_c(T.cat([h_new, b_new], dim=-1))

        return {
            's': s,
            'a': (h_new, b_new),
            'g': g,
            'c': c,
        }


class ReadoutModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ReadoutModule, self).__init__()
        h_dims = kwargs['h_dims']
        n_classes = kwargs['n_classes']
        self.f_out = nn.Sequential(
            nn.Linear(h_dims, h_dims),
            nn.ReLU(),
            nn.Linear(h_dims, n_classes),
        )

    def forward(self, node_states, edge_states):
        #att_weights = F.normalize(T.cat([s['c'] for s in node_states], 1), p=1)[..., None]
        #hids = T.stack([s['a'][0] for s in node_states], 1)
        #hids = (hids * att_weights).sum(dim=1)
        hids = T.stack([s['a'][0] for s in node_states], 1)
        return self.f_out(hids)

class TopDownNet(nn.Module):
    def __init__(self,
                 h_dims=128,
                 n_classes=10,
                 filters=[16, 32, 64, 128, 256],
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 glimpse_type='gaussian',
                 glimpse_size=(15, 15),
                 cnn='cnn'
    ):
        from networkx.algorithms.traversal.breadth_first_search import bfs_edges
        nn.Module.__init__(self)
        t = nx.balanced_tree(1, 2)
        self.G = DGLGraph(t)
        self.root = 0
        #self.walk_list = bfs_edges(t, self.root)
        self.walk_list = [(0, 1), (1, 2)]
        self.h_dims = h_dims
        self.n_classes = n_classes

        self.update_module = UpdateModule(
            h_dims=h_dims,
            n_classes=n_classes,
            filters=filters,
            kernel_size=kernel_size,
            final_pool_size=final_pool_size,
            glimpse_type=glimpse_type,
            glimpse_size=glimpse_size,
            cnn='cnn',
        )
        self.message_module = MessageModule(
            h_dims=h_dims,
            g_dims=self.update_module.glimpse.att_params
        ) 
        self.readout_module = ReadoutModule(
            h_dims=h_dims,
            n_classes=n_classes,
        )

        self.G.register_message_func(self.message_module)
        self.G.register_update_func(self.update_module)
        self.G.register_readout_func(self.readout_module)

    def forward(self, x):
        batch_size = x.shape[0]
        g_dims = self.update_module.glimpse.att_params

        self.update_module.set_image(x)
        zero_tensor_x = lambda r, c: \
            x.new(r, c).zero_()

        init_states = {
            's': zero_tensor_x(batch_size, self.h_dims),
            'a': (
                zero_tensor_x(batch_size, self.h_dims),
                zero_tensor_x(batch_size, g_dims),
            ),
            'g': None,
            'c': zero_tensor_x(batch_size, 1),
        }

        for n in self.G.nodes():
            self.G.node[n].update(init_states)

        self.G.recvfrom(self.root, [])   # Update root node
        self.G.propagate(self.walk_list)
        return self.G.readout()

"""
Copied from Andy
"""
class Net(skorch.NeuralNet):
    def __init__(self, **kwargs):
        self.reg_coef_ = kwargs.get('reg_coef', 1e-4)

        del kwargs['reg_coef']
        skorch.NeuralNet.__init__(self, **kwargs)

    def initialize_criterion(self):
        # Overriding this method to skip initializing criterion as we don't use it.
        pass

    def get_split_datasets(self, X, y=None, **fit_params):
        # Overriding this method to use our own dataloader to change the X
        # in signature to (train_dataset, valid_dataset)
        X_train, X_valid = X
        train = self.get_dataset(X_train, None)
        valid = self.get_dataset(X_valid, None)
        return train, valid

    def train_step(self, Xi, yi, **fit_params):
        step = skorch.NeuralNet.train_step(self, Xi, yi, **fit_params)
        loss = step['loss']
        y_pred = step['y_pred']
        acc = self.get_loss(y_pred, yi, training=False)
        self.history.record_batch('max_param', max(p.abs().max().item() for p in self.module_.parameters()))
        self.history.record_batch('acc', acc.item())
        return {
                'loss': loss,
                'y_pred': y_pred,
        }

    def get_loss(self, y_pred, y_true, X=None, training=False):
        #batch_size, _ = y_pred.shape
        batch_size, n_steps, _ = y_pred.shape
        if training:
            y_true = y_true[:, None].expand(batch_size, n_steps)
            return F.cross_entropy(
                    y_pred.reshape(batch_size * n_steps, -1),
                    y_true.reshape(-1)
                    )
            #return F.cross_entropy(
            #    y_pred, y_true
            #)
        else:
            y_prob, y_cls = y_pred.max(-1)
            _, y_prob_maxind = y_prob.max(-1)
            y_cls_final = y_cls.gather(1, y_prob_maxind[:, None])[:, 0]
            return (y_cls_final == y_true).sum()
            #y_prob, y_cls = y_pred.max(-1)
            #return (y_cls == y_true).sum()

def init_canvas(n_nodes):
    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(16, 8)
    return fig, ax

def display_image(fig, ax, i, im):
    im = im.detach().cpu().numpy().transpose(1, 2, 0)
    ax[i // 4, i % 4].imshow(im, cmap='gray', vmin=0, vmax=1)

class Dump(skorch.callbacks.Callback):
    def initialize(self):
        self.epoch = 0
        self.batch = 0
        self.correct = 0
        self.total = 0
        self.best_acc = 0
        self.nviz = 0
        return self

    def on_epoch_begin(self, net, **kwargs):
        self.epoch += 1
        self.batch = 0
        self.correct = 0
        self.total = 0
        self.nviz = 0

    def on_batch_end(self, net, **kwargs):
        self.batch += 1
        if kwargs['training']:
            pass
        else:
            self.correct += kwargs['loss'].item()
            self.total += kwargs['X'].shape[0]

            # Visualize
            if self.nviz < 10:
                n_nodes = len(net.module_.G.nodes)
                fig, ax = init_canvas(n_nodes)

                for i, n in enumerate(net.module_.G.nodes):
                    repr_ = net.module_.G.nodes[n]
                    g = repr_['g']
                    if g is None:
                        continue
                    display_image(
                        fig,
                        ax,
                        i,
                        g[0],
                    )

                wm.display_mpl_figure(fig, win='viz{}'.format(self.nviz))
                self.nviz += 1

    def on_epoch_end(self, net, **kwargs):
        print('@', self.epoch, self.correct, '/', self.total)
        acc = self.correct / self.total
        if self.best_acc < acc:
            self.best_acc = acc
            net.history.record('acc_best', acc)
        else:
            net.history.record('acc_best', None)

"""
Copied from Andy's code.
"""
def data_generator(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    for _x, _y, _B in dataloader:
        x = _x[:, None].expand(_x.shape[0], 3, _x.shape[1], _x.shape[2]).float() / 255.
        y = _y.squeeze(1)
        yield cuda(x), cuda(y)

if __name__ == "__main__":
    from datasets import MNISTMulti
    from torch.utils.data import DataLoader

    mnist_train = MNISTMulti('.', n_digits=1, backrand=0, image_rows=200, image_cols=200, download=True)
    mnist_valid = MNISTMulti('.', n_digits=1, backrand=0, image_rows=200, image_cols=200, mode='valid')

    reg_coef = 0
    print("Trying ref coef: {}". format(reg_coef))
    net = Net(
        module=TopDownNet,
        criterion=None,
        max_epochs=50,
        reg_coef=reg_coef,
        optimizer=T.optim.RMSprop,
        lr=1e-5,
        batch_size=batch_size,
        device='cuda',
        callbacks=[
            Dump(),
            skorch.callbacks.Checkpoint(monitor='acc_best'),
            skorch.callbacks.ProgressBar(postfix_keys=['train_loss', 'valid_loss', 'acc', 'reg']),
            skorch.callbacks.GradientNormClipping(0.01)
        ],
        iterator_train=data_generator,
        iterator_train__shuffle=True,
        iterator_valid=data_generator,
        iterator_valid__shuffle=False
    )

    net.partial_fit((mnist_train, mnist_valid), epochs=500)
