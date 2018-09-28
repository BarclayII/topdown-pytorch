"""
MNIST Clutter
Modified from https://github.com/deepmind/mnist-cluttered
"""

import torch as T
import numpy as np

M = {}

def selectSamples(examples, nSamples):
    nExamples = examples.size(0)
    samples = []
    for _ in range(nSamples):
        samples.append(examples[np.random.randint(nExamples)])
    return samples

def placeDistractors(config, patch, examples):
    distractors = selectSamples(examples, config['num_dist'])
    dist_w = config['dist_w']
    megapatch_w = config['megapatch_w']

    for d_patch in distractors:
        t_y = np.random.randint((megapatch_w - dist_w))
        t_x = np.random.randint((megapatch_w - dist_w))
        s_y = np.random.randint((d_patch.size(0) - dist_w))
        s_x = np.random.randint((d_patch.size(1) - dist_w))
        patch[t_y: t_y + dist_w, t_x: t_x + dist_w] += \
            d_patch[s_y: s_y + dist_w, s_x: s_x + dist_w]
        patch.clamp_(max=1)

def placeSpriteRandomly(obs, sprite, border, idx=0, n=1):
    assert obs.dim() == 2, 'expecting an image'
    assert sprite.dim() == 2, 'expecting a sprite'
    h = obs.size(0)
    w = obs.size(1)
    spriteH = sprite.size(0)
    spriteW = sprite.size(1)

    y = np.random.randint(border, h - spriteH - border)
    x = np.random.randint(border, w - spriteW - border)

    subTensor = obs[y: y + spriteH, x: x + spriteW]
    subTensor += sprite
    subTensor.clamp_(0, 1)

class ClutteredMNIST(object):
    def __init__(self, dataset, **kwargs):
        import os
        self.config = {
            'megapatch_w': 100,
            'num_dist': 8,
            'dist_w': 8,
            'border': 0,
            'nDigits': 1,
        }
        for k, v in kwargs.items():
            if k not in self.config:
                assert "unsupported params"
            self.config[k] = v

        self.data, self.labels = dataset
        self.nExamples = self.data.size(0)
        self._step = self.nExamples
        self._perm = None

    def __iter__(self):
        return self

    def __next__(self):
        obs = T.zeros(self.config['megapatch_w'], self.config['megapatch_w'])
        lbls = T.LongTensor(self.config['nDigits'])
        placeDistractors(self.config, obs, self.data)
        for i in range(self.config['nDigits']):
            self._step += 1
            if self._step >= self.nExamples:
                self._perm = T.randperm(self.nExamples)
                self._step = 0

            sprite = self.data[self._perm[self._step]]
            placeSpriteRandomly(obs, sprite, self.config['border'], idx=i, n=self.config['nDigits'])

            selectedDigit = self.labels[self._perm[self._step]]
            lbls[i] = selectedDigit

        return obs, lbls

    def get_bunch(self, size):
        data = T.zeros(size, self.config['megapatch_w'], self.config['megapatch_w'])
        labels = T.LongTensor(size, self.config['nDigits'])
        for i in range(size):
            data[i], labels[i] = self.__next__()
        return data * 255, labels

if __name__ == '__main__':
    data = ClutteredMNIST()
    for img, lbl in data:
        import matplotlib.pyplot as plt
        plt.imshow(img.numpy())
        plt.show()
        print(lbl)
        break
    x, y = data.get_bunch(10000)
    print(x.shape, y.shape)
