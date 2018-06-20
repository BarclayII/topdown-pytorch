from topdown import *
import tensorflow as tf
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from datasets import MNISTMulti
import torch as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tvmodels
from util import USE_CUDA, cuda
from viz import VisdomWindowManager
from util import addbox
import os

baseline = os.getenv('BASELINE', 0)

wm = VisdomWindowManager(port=10248)

def data_generator(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)
    for _x, _y, _B in dataloader:
        x = _x[:, None].expand(_x.shape[0], 3, _x.shape[1], _x.shape[2]).float() / 255.
        y = _y.squeeze(1)
        yield cuda(x), cuda(y)

mnist = MNISTMulti('.', n_digits=1, backrand=0, image_rows=200, image_cols=200, download=False, mode='test')
loader = data_generator(mnist, 1, False)

class TemporaryModule(T.nn.Module):
    '''
    hacks around the restriction from cleverhans that requires a 2D logits tensor
    '''
    def __init__(self, model):
        T.nn.Module.__init__(self)
        self.model = model

    def forward(self, x):
        y = self.model(x)
        if y.dim() == 3:
            return y.squeeze(1)
        else:
            return y

#model = cuda(DFSGlimpseSingleObjectClassifier())
model = cuda(tvmodels.ResNet(tvmodels.resnet.BasicBlock, [2, 2, 2, 2], 10))
model.load_state_dict(T.load('model.pt'))

s = tf.Session()
x_op = tf.placeholder(tf.float32, shape=(None, 3, 200, 200))

tf_model_fn = convert_pytorch_model_to_tf(cuda(TemporaryModule(model)))
cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

fgsm_op = FastGradientMethod(cleverhans_model, sess=s)
fgsm_params = {'eps': 0.01, 'clip_min': 0, 'clip_max': 1}
adv_x_op = fgsm_op.generate(x_op, **fgsm_params)
adv_preds_op = tf_model_fn(adv_x_op)
preds_op = tf_model_fn(x_op)

total = 0
correct = 0
adv_correct = 0
nviz = 0

for xs, ys in loader:
    y = ys[0].item()
    preds = s.run(preds_op, feed_dict={x_op: xs})[0]

    if not baseline:
        ys_by_node = [F.softmax(model.G.nodes[v]['y'], -1)[0].detach().cpu().numpy()
                      for v in model.G.nodes]
        gs_by_node = [model.G.nodes[v]['g'][0] for v in model.G.nodes]
        bs_by_node = [model.update_module.glimpse.rescale(model.G.nodes[v]['b_fix'][0], False)[0].detach().cpu().numpy()
                      for v in model.G.nodes]

    adv_xs, adv_preds = s.run([adv_x_op, adv_preds_op], feed_dict={x_op: xs})
    adv_preds = adv_preds[0]

    if not baseline:
        adv_ys_by_node = [F.softmax(model.G.nodes[v]['y'], -1)[0].detach().cpu().numpy()
                          for v in model.G.nodes]
        adv_gs_by_node = [model.G.nodes[v]['g'][0] for v in model.G.nodes]
        adv_bs_by_node = [model.update_module.glimpse.rescale(model.G.nodes[v]['b_fix'][0], False)[0].detach().cpu().numpy()
                          for v in model.G.nodes]

    preds_cls = preds.argmax()
    adv_preds_cls = adv_preds.argmax()
    #print(y, preds_cls, adv_preds_cls)
    total += 1
    correct += 1 if preds_cls == y else 0
    adv_correct += 1 if adv_preds_cls == y else 0
    if total % 500 == 0:
        print(total, correct, adv_correct)

        if not baseline:
            fig, ax = init_canvas(8)

            display_image(fig, ax, 0, xs[0], 'original image')
            display_image(fig, ax, 4, adv_xs[0], 'adversarial image')
            for i in range(3):
                cls_by_node = ys_by_node[i].argmax()
                display_image(fig, ax, 1 + i, gs_by_node[i],
                              'y=%d (%.2f)' % (cls_by_node, ys_by_node[i][cls_by_node]) +
                              'y*=%d' % y)
                addbox(ax[0, 0], bs_by_node[i][:4] * 200, 'red', 5 + 2 * i)

                adv_cls_by_node = adv_ys_by_node[i].argmax()
                display_image(fig, ax, 5 + i, adv_gs_by_node[i],
                              'y=%d (%.2f)' % (adv_cls_by_node, adv_ys_by_node[i][adv_cls_by_node]) +
                              'y*=%d' % y)
                addbox(ax[1, 0], adv_bs_by_node[i][:4] * 200, 'yellow', 5 + 2 * i)

            wm.display_mpl_figure(fig, win='viz{}'.format(nviz))
            nviz += 1

print(total, correct, adv_correct)
