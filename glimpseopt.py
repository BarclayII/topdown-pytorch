import torch as T
import torch.nn.functional as F
from torchvision.datasets import MNIST
from datasets import MNISTMulti
from topdown import *
from util import USE_CUDA, cuda
import tqdm
import pickle
import numpy as np

#mnist = MNIST('.', download=True)
mnist = MNISTMulti('.', n_digits=1, backrand=0, image_rows=200, image_cols=200, download=True)

glimpse = MultiscaleGlimpse(glimpse_type='gaussian', glimpse_size=(15, 15))
module = cuda(CNN(cnn='cnn', input_size=(15, 15), h_dims=128, n_classes=10, kernel_size=(3, 3), final_pool_size=(1, 1), filters=[16, 32, 64, 128, 256], pred=True, in_channels=9))
seq = T.nn.Sequential(glimpse, module)
seq.load_state_dict(T.load('cnntest.pt'))


rec = []

for i in range(mnist.train_data.shape[0]):
    x = cuda(mnist.train_data[i:i+1, None].repeat(1, 3, 1, 1).float() / 255.)
    y = cuda(mnist.train_labels[i:i+1, 0])
    b = cuda(T.zeros(1, 6))
    b.requires_grad = True

    rec.append({})
    rec[-1]['i'] = i
    rec[-1]['x'] = x.cpu().numpy()
    rec[-1]['y'] = y.cpu().numpy()
    rec[-1]['b'] = mnist.train_locs[i, 0].cpu().numpy()
    rec[-1]['bh'] = []
    rec[-1]['loss'] = []
    rec[-1]['pred'] = []
    rec[-1]['bgrad'] = []

    with tqdm.trange(10000) as tqdm_obj:
        for _ in tqdm_obj:
            #br, _ = glimpse.rescale(b[:, None], False)
            #g = glimpse(x, br)[:, 0]
            br, _ = glimpse.glimpse.rescale(b[:, None], False)
            g = glimpse(x, b)
            cls = module.forward(g)
            loss = F.cross_entropy(cls, y)

            rec[-1]['bh'].append(br[:, 0].detach().cpu().numpy())
            rec[-1]['pred'].append(cls.detach().cpu().numpy())
            rec[-1]['loss'].append(loss.detach().cpu().numpy())

            if b.grad is not None:
                b.grad.zero_()
            loss.backward()
            rec[-1]['bgrad'].append(b.grad.cpu().numpy())
            b = (b - 1e-1 * b.grad).detach()
            b.requires_grad = True
            tqdm_obj.set_postfix(loss=np.asscalar(rec[-1]['loss'][-1]))

    rec[-1]['bh'] = np.array(rec[-1]['bh'])
    rec[-1]['loss'] = np.array(rec[-1]['loss'])
    rec[-1]['pred'] = np.array(rec[-1]['pred'])
    rec[-1]['bgrad'] = np.array(rec[-1]['bgrad'])

    if i % 100 == 99:
        with open('glimpseopt/%05d.pkl' % i, 'wb') as f:
            pickle.dump(rec, f)
        rec = []
