from topdown import *
from datasets import MNISTMulti
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV
import skorch
from viz import VisdomWindowManager
from util import USE_CUDA, cuda
import torchvision.models as tvmodels
import os

batch_size = 32
wm = VisdomWindowManager(port=10248)

baseline = os.getenv('BASELINE', 0)

class NetWithTwoDatasets(skorch.NeuralNet):
    def get_split_datasets(self, X, y=None, **fit_params):
        # Overriding this method to use our own dataloader to change the X
        # in signature to (train_dataset, valid_dataset)
        X_train, X_valid = X
        train = self.get_dataset(X_train, None)
        valid = self.get_dataset(X_valid, None)
        return train, valid


class Net(NetWithTwoDatasets):
    def __init__(self, **kwargs):
        self.reg_coef_ = kwargs.get('reg_coef', 1e-4)

        del kwargs['reg_coef']
        skorch.NeuralNet.__init__(self, **kwargs)

    def initialize_criterion(self):
        # Overriding this method to skip initializing criterion as we don't use it.
        pass

    def train_step(self, Xi, yi, **fit_params):
        step = skorch.NeuralNet.train_step(self, Xi, yi, **fit_params)
        dbs = [self.module_.G.nodes[v]['db'] for v in self.module_.G.nodes]
        reg = self.reg_coef_ * sum(db.norm(2, 1).mean() for db in dbs if db is not None)
        loss = step['loss'] + reg
        y_pred = step['y_pred']
        acc = self.get_loss(y_pred, yi, training=False)
        self.history.record_batch('max_param', max(p.abs().max().item() for p in self.module_.parameters()))
        self.history.record_batch('acc', acc.item())
        self.history.record_batch('reg', reg.item())
        return {
                'loss': loss,
                'y_pred': y_pred,
                }

    def get_loss(self, y_pred, y_true, X=None, training=False):
        batch_size, n_steps, _ = y_pred.shape
        if training:
            #return F.cross_entropy(y_pred, y_true)
            y_true = y_true[:, None].expand(batch_size, n_steps)
            return F.cross_entropy(
                    y_pred.reshape(batch_size * n_steps, -1),
                    y_true.reshape(-1)
                    )
        else:
            y_prob, y_cls = y_pred.max(-1)
            _, y_prob_maxind = y_prob.max(-1)
            y_cls_final = y_cls.gather(1, y_prob_maxind[:, None])[:, 0]
            return (y_cls_final == y_true).sum()


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
            #print('#', self.epoch, self.batch, kwargs['loss'], kwargs['valid_loss'])
            pass
        else:
            self.correct += kwargs['loss'].item()
            self.total += kwargs['X'].shape[0]

            if self.nviz < 10:
                n_nodes = len(net.module_.G.nodes)
                fig, ax = init_canvas(n_nodes)
                #a = T.stack([net.module_.G.nodes[v]['a'] for v in net.module_.G.nodes], 1)
                #a = F.softmax(a, 1).detach().cpu().numpy()
                y = T.stack([net.module_.G.nodes[v]['y'] for v in net.module_.G.nodes], 1)
                y = F.softmax(y, -1)
                y_val, y = y.max(-1)
                for i, n in enumerate(net.module_.G.nodes):
                    repr_ = net.module_.G.nodes[n]
                    g = repr_['g']
                    if g is None:
                        continue
                    b, _ = net.module_.update_module.glimpse.rescale(repr_['b'], False)
                    display_image(
                            fig,
                            ax,
                            i,
                            g[0],
                            np.array_str(
                                b[0].detach().cpu().numpy(),
                                precision=2, suppress_small=True) +
                            #'a=%.2f' % a[0, i, 0]
                            'y=%d (%.2f)' % (y[0, i], y_val[0, i]) +
                            'y*=%d' % kwargs['y'][0]
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


def data_generator(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    for _x, _y, _B in dataloader:
        x = _x[:, None].expand(_x.shape[0], 3, _x.shape[1], _x.shape[2]).float() / 255.
        y = _y.squeeze(1)
        yield cuda(x), cuda(y)

mnist_train = MNISTMulti('.', n_digits=1, backrand=0, image_rows=200, image_cols=200, download=True)
mnist_valid = MNISTMulti('.', n_digits=1, backrand=0, image_rows=200, image_cols=200, download=False, mode='valid')

for reg_coef in [0]:
    print('Trying reg coef', reg_coef)

    net_kwargs = dict(
            module=DFSGlimpseSingleObjectClassifier,
            criterion=None,
            max_epochs=50,
            optimizer=T.optim.RMSprop,
            #optimizer__weight_decay=1e-4,
            lr=1e-5,
            batch_size=batch_size,
            device='cuda' if USE_CUDA else 'cpu',
            callbacks=[
                skorch.callbacks.ProgressBar(postfix_keys=['train_loss', 'valid_loss', 'acc', 'reg']),
                skorch.callbacks.GradientNormClipping(0.01),
                #skorch.callbacks.LRScheduler('ReduceLROnPlateau'),
                ],
            iterator_train=data_generator,
            iterator_train__shuffle=True,
            iterator_valid=data_generator,
            iterator_valid__shuffle=False,
            )
    if baseline:
        net_kwargs.update(dict(
            criterion=T.nn.CrossEntropyLoss,
            module=tvmodels.ResNet,
            module__block=tvmodels.resnet.BasicBlock,
            module__layers=[2, 2, 2, 2],
            module__num_classes=10,
            ))
        net_kwargs['callbacks'].insert(0, skorch.callbacks.Checkpoint())
        net = NetWithTwoDatasets(**net_kwargs)
    else:
        net_kwargs['callbacks'].insert(0, skorch.callbacks.Checkpoint(monitor='acc_best'))
        net_kwargs['callbacks'].insert(0, Dump())
        net_kwargs['reg_coef'] = reg_coef
        net = Net(**net_kwargs)

    #net.fit((mnist_train, mnist_valid), pretrain=True, epochs=50)
    net.partial_fit((mnist_train, mnist_valid), epochs=500)
