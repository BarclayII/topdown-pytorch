import networkx as nx
from glimpse import create_glimpse
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as MODELS
import torch.nn.init as INIT
import numpy as np
from dgl.graph import DGLGraph
import matplotlib.pyplot as plt
from util import *

def dfs_walk(tree, curr, l):
    if len(tree.succ[curr]) == 0:
        return
    else:
        for n in tree.succ[curr]:
            l.append((curr, n))
            dfs_walk(tree, n, l)
            l.append((n, curr))

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
        #INIT.xavier_uniform_(module.weight)
        INIT.constant_(module.bias, 0)
        cnn_list.append(module)
        if i < len(filters) - 1:
            cnn_list.append(nn.LeakyReLU())
    cnn_list.append(nn.AdaptiveMaxPool2d(final_pool_size))

    return nn.Sequential(*cnn_list)

def build_resnet_cnn(**config):
    n_layers = config['n_layers']
    final_pool_size = config['final_pool_size']

    resnet = MODELS.resnet18(pretrained=False)
    cnn_list = list(resnet.children())[0:n_layers]
    cnn_list.append(nn.AdaptiveMaxPool2d(final_pool_size))

    return nn.Sequential(*cnn_list)



def init_canvas(n_nodes):
    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(16, 8)
    return fig, ax


def display_image(fig, ax, i, im, title):
    if T.is_tensor(im):
        im = im.detach().cpu().numpy()
    im = im.transpose(1, 2, 0)
    ax[i // 4, i % 4].imshow(im, cmap='gray', vmin=0, vmax=1)
    ax[i // 4, i % 4].set_title(title, fontsize=6)

class CNN(nn.Module):
    def __init__(self, **config):
        nn.Module.__init__(self)

        cnn = config['cnn']
        input_size = config['input_size']
        final_pool_size = config['final_pool_size']
        h_dims = config['h_dims']
        n_classes = config['n_classes']
        in_channels = config.get('in_channels', 3)
        if cnn == 'resnet':
            n_layers = config['n_layers']
            self.cnn = build_resnet_cnn(
                    n_layers=n_layers,
                    final_pool_size=final_pool_size,
                    in_channels=in_channels,
                    )
            self.net_h = nn.Linear(128 * np.prod(final_pool_size), h_dims)
        else:
            filters = config['filters']
            kernel_size = config['kernel_size']
            self.cnn = build_cnn(
                    filters=filters,
                    kernel_size=kernel_size,
                    final_pool_size=final_pool_size,
                    in_channels=in_channels,
                    )
            self.net_h = nn.Linear(filters[-1] * np.prod(final_pool_size), h_dims)

        self.net_p = nn.Sequential(
                nn.Linear(h_dims, h_dims),
                nn.ReLU(),
                nn.Linear(h_dims, n_classes),
                )
        self.input_size = input_size
        self.pred = config.get('pred', False)

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.net_h(self.cnn(x).view(batch_size, -1))
        return self.net_p(h) if self.pred else h


class MultiscaleGlimpse(nn.Module):
    multiplier = cuda(T.FloatTensor(
            [[1, 1, 1, 1, 1, 1],
             [1, 1, 2, 2, 2, 2],
             [1, 1, 3, 3, 3, 3]]
            ))

    def __init__(self, **config):
        nn.Module.__init__(self)

        glimpse_type = config['glimpse_type']
        self.glimpse_size = config['glimpse_size']
        self.n_glimpses = config['n_glimpses']
        self.glimpse = create_glimpse(glimpse_type, self.glimpse_size)

        self.multiplier = cuda(T.cat([
            T.ones(self.n_glimpses, 2),
            T.arange(1, self.n_glimpses + 1).view(-1, 1).repeat(1, 4),
            ], 1))

    def forward(self, x, b=None):
        batch_size, n_channels = x.shape[:2]
        if b is None:
            # defaults to full canvas
            b = x.new(batch_size, self.glimpse.att_params).zero_()
        b, _ = self.glimpse.rescale(b[:, None], False)
        b = b.repeat(1, self.n_glimpses, 1) * self.multiplier[None]
        g = self.glimpse(x, b).view(
                batch_size, self.n_glimpses * n_channels, self.glimpse_size[0], self.glimpse_size[1])
        return g

class MessageModule(nn.Module):
    def forward(self, src, dst, edge):
        h, b_next = [src[k] for k in ['h', 'b_next']]
        return h, b_next

class UpdateModule(nn.Module):
    """
    UpdateModule:

    Returns:
        h: new state
        b: new bounding box
        a: attention (for readout)
        y: prediction
    """
    def __init__(self, **config):
                 #h_dims=128,
                 #n_classes=10,
                 #steps=5,
                 #filters=[16, 32, 64, 128, 256],
                 #kernel_size=(3, 3),
                 #final_pool_size=(2, 2),
                 #glimpse_type='gaussian',
                 #glimpse_size=(15, 15),
                 #cnn='resnet'
                 #):
        super(UpdateModule, self).__init__()
        glimpse_type = config['glimpse_type']
        glimpse_size = config['glimpse_size']
        self.glimpse = create_glimpse(glimpse_type, glimpse_size)

        h_dims = config['h_dims']
        n_classes = config['n_classes']
        self.net_b = nn.Sequential(
                nn.Linear(h_dims, h_dims),
                nn.ReLU(),
                nn.Linear(h_dims, self.glimpse.att_params),
                )
        self.net_y = nn.Sequential(
                nn.Linear(h_dims, h_dims),
                nn.ReLU(),
                nn.Linear(h_dims, n_classes),
                )
        self.net_a = nn.Sequential(
                nn.Linear(h_dims, h_dims),
                nn.ReLU(),
                nn.Linear(h_dims, 1),
                )

        self.h_to_h = nn.GRUCell(h_dims * 2, h_dims)
        #INIT.orthogonal_(self.h_to_h.weight_hh)

        self.cnn = config['cnn']

        self.max_recur = config.get('max_recur', 1)
        self.h_dims = h_dims

    def set_image(self, x):
        self.x = x

    def forward(self, node_state, message):
        h, b, y, b_fix = [node_state[k] for k in ['h', 'b', 'y', 'b_fix']]
        batch_size = h.shape[0]
        new_node = True

        if len(message) == 0:
            h_m_avg = h.new(batch_size, self.h_dims).zero_()
        else:
            h_m, b_next = zip(*message)
            h_m_avg = T.stack(h_m).mean(0)
            if b_fix is not None:
                new_node = False
            b = T.stack(b_next).mean(0) if new_node else b_fix

        b_new = b_fix = b
        h_new = h

        for i in range(self.max_recur):
            b_rescaled, _ = self.glimpse.rescale(b_new[:, None], False)
            g = self.glimpse(self.x, b_rescaled)[:, 0]
            h_in = T.cat([self.cnn(g), h_m_avg], -1)
            h_new = self.h_to_h(h_in, h_new)

            db = self.net_b(h_new)
            dy = self.net_y(h_new)
            b_new = b + db
            y_new = y + dy
            a_new = self.net_a(h_new)

        b_rescaled, _ = self.glimpse.rescale(b[:, None], False)
        b_new_rescaled, _ = self.glimpse.rescale(b_new[:, None], False)
        b_rescaled = b_rescaled[:, 0]
        b_new_rescaled = b_new_rescaled[:, 0]

        if new_node:
            new_area = area(b_new_rescaled)
            bbox_penalty = (new_area - intersection(b_rescaled, b_new_rescaled) + 1e-6) / (new_area + 1e-6)
            assert (bbox_penalty < -1e-4).sum().item() == 0
            bbox_penalty = bbox_penalty.clamp(min=0)
        else:
            bbox_penalty = None

        return {'h': h_new, 'b': b, 'b_next': b_new, 'a': a_new, 'y': y_new, 'g': g, 'b_fix': b_fix,
                'db': db, 'bbox_penalty': bbox_penalty}

def update_local():
    pass

class ReadoutModule(nn.Module):
    '''
    Returns the logits of classes
    '''
    def __init__(self, *args, **kwarg):
        super(ReadoutModule, self).__init__()
        self.y = nn.Linear(kwarg['h_dims'], kwarg['n_classes'])

    def forward(self, nodes_state, edge_states, pretrain=False):
        if pretrain:
            assert len(nodes_state) == 1        # root only
            h = nodes_state[0]['h']
            y = self.y(h)
        else:
            #h = T.stack([s['h'] for s in nodes_state], 1)
            #a = F.softmax(T.stack([s['a'] for s in nodes_state], 1), 1)
            #b_of_h = T.sum(a * h, 1)
            #b_of_h = h[:, -1]
            #y = self.y(b_of_h)
            y = nodes_state[0]['y'][:, None]
            #y = T.stack([s['y'] for s in nodes_state], 1)
        return y

class DFSGlimpseSingleObjectClassifier(nn.Module):
    def __init__(self,
                 h_dims=128,
                 n_classes=10,
                 filters=[16, 32, 64, 128, 256],
                 kernel_size=(3, 3),
                 final_pool_size=(2, 2),
                 glimpse_type='gaussian',
                 glimpse_size=(15, 15),
                 cnn='cnn',
                 cnn_file='cnn.pt',
                 ):
        nn.Module.__init__(self)

        #self.T_MAX_RECUR = kwarg['steps']

        t = nx.balanced_tree(2, 2)
        t_uni = nx.bfs_tree(t, 0)
        self.G = DGLGraph(t)
        self.root = 0
        self.h_dims = h_dims
        self.n_classes = n_classes

        self.message_module = MessageModule()
        self.G.register_message_func(self.message_module) # default: just copy

        cnnmodule = CNN(
                cnn=cnn,
                n_layers=6,
                h_dims=h_dims,
                n_classes=n_classes,
                final_pool_size=final_pool_size,
                filters=filters,
                kernel_size=kernel_size,
                input_size=glimpse_size,
                )
        if cnn_file is not None:
            cnnmodule.load_state_dict(T.load(cnn_file))

        #self.update_module = UpdateModule(h_dims, n_classes, glimpse_size)
        self.update_module = UpdateModule(
            glimpse_type=glimpse_type,
            glimpse_size=glimpse_size,
            cnn=cnnmodule,
            max_recur=1,    # T_MAX_RECUR
            n_classes=n_classes,
            h_dims=h_dims,
        )
        self.G.register_update_func(self.update_module)

        self.readout_module = ReadoutModule(h_dims=h_dims, n_classes=n_classes)
        self.G.register_readout_func(self.readout_module)

        #self.walk_list = [(0, 1), (1, 2), (2, 1), (1, 0)]
        self.walk_list = []
        dfs_walk(t_uni, self.root, self.walk_list)

    def forward(self, x, pretrain=False):
        batch_size = x.shape[0]

        self.update_module.set_image(x)
        init_states = {
            'h': x.new(batch_size, self.h_dims).zero_(),
            'b': x.new(batch_size, self.update_module.glimpse.att_params).zero_(),
            'b_next': x.new(batch_size, self.update_module.glimpse.att_params).zero_(),
            'a': x.new(batch_size, 1).zero_(),
            'y': x.new(batch_size, self.n_classes).zero_(),
            'g': None,
            'b_fix': None,
            'db': None,
            }
        for n in self.G.nodes():
            self.G.node[n].update(init_states)

        #TODO: the following two lines is needed for single object
        #TODO: but not useful or wrong for multi-obj
        self.G.recvfrom(self.root, [])

        if pretrain:
            return self.G.readout([self.root], pretrain=True)
        else:
            #for u, v in self.walk_list:
            #    self.G.update_by_edge((u, v))
                # update local should be inside the update module
                #for i in self.T_MAX_RECUR:
                #    self.G.update_local(u)
            self.G.propagate(self.walk_list)
            return self.G.readout('all', pretrain=False)
