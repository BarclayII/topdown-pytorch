
import torch as th
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from tqdm import tqdm
import os
import cv2
import numpy as np

def visualize(fig):
#    import matplotlib
#    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imshow(fig, cmap='gray')
    plt.show()

class MNISTScale(Dataset):
    dir_ = 'scale'
    attr_prefix = {'train': 'training', 'valid': 'valid', 'test': 'test'}
    n_classes = 1000
    mnist_ori_row = 28
    mnist_ori_col = 28

    @property
    def _meta(self):
        return '%d-%d-%.2f.pt' % (
            self.image_rows,
            self.image_cols,
            self.scale)

    @property
    def training_file(self):
        return os.path.join(self.dir_, 'training-' + self._meta)

    @property
    def valid_file(self):
        return os.path.join(self.dir_, 'valid-' + self._meta)

    @property
    def test_file(self):
        return os.path.join(self.dir_, 'test-' + self._meta)

    def __init__(self,
                 root,
                 mode='train',
                 image_rows=28,
                 image_cols=28,
                 download=False,
                 scale=1):
        if scale > 1.0:
            raise ValueError("Scale {} exceeds limit 1".format(scale))
        self.mode = mode
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.scale = scale

        if os.path.exists(self.dir_):
            if os.path.isfile(self.dir_):
                raise NotADirectoryError(self.dir_)
            elif os.path.exists(getattr(self, self.attr_prefix[mode] + '_file')):
                with open(getattr(self, self.attr_prefix[mode] + '_file'), 'rb') as f:
                    data = th.load(f)
                for k in data:
                    setattr(self, mode + '_' + k, data[k])
                self.size = getattr(self, mode + '_data').size()[0]
                return
        elif not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

        valid_src_size = 10000

        np.random.seed(1111)
        for _mode in ['train', 'valid', 'test']:
            _train = _mode is not 'test'
            mnist = MNIST(root, _train, download=download)

            if _mode == 'train':
                src_data = mnist.train_data[:-valid_src_size]
                src_labels = mnist.train_labels[:-valid_src_size]
            elif _mode == 'valid':
                src_data = mnist.train_data[-valid_src_size:]
                src_labels = mnist.train_labels[-valid_src_size:]
            elif _mode == 'test':
                src_data = mnist.test_data
                src_labels = mnist.test_labels

            data = th.ByteTensor(len(src_data), image_rows, image_cols).zero_()
            labels = th.LongTensor(len(src_labels), 1).zero_()
            resized_rows = int(self.mnist_ori_row * self.scale)
            resized_cols = int(self.mnist_ori_col * self.scale)
            if self.image_rows == resized_rows:
                float_row = 0
            else:
                float_row = np.random.randint(self.image_rows - resized_rows)

            if self.image_cols == resized_cols:
                float_col = 0
            else:
                float_col = np.random.randint(self.image_cols - resized_cols)

            for idx, cur_data in tqdm(enumerate(src_data)):
                resized_data = th.from_numpy(
                    cv2.resize(
                        cur_data.numpy(),
                        (resized_rows, resized_cols))
                )
                data[idx, float_row: float_row + resized_rows, float_col: float_col + resized_cols] = resized_data
                labels[idx] = src_labels[idx]

            th.save({
                'data': data,
                'labels': labels,
            }, getattr(self, self.attr_prefix[_mode] + '_file'))

            if _mode == mode:
                setattr(self, mode + '_data', data)
                setattr(self, mode + '_labels', labels)
                self.size = data.size()[0]

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return tuple(getattr(self, self.mode + '_' + k)[i] for k in ['data', 'labels'])

if __name__ == '__main__':
    datasets = [None] * 10
    for idx, scale in enumerate(np.linspace(0.1, 1, 10)):
        datasets[idx] = MNISTScale('..', scale=scale)#, download=True)
