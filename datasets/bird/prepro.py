import torch as T
import scipy.io
import os
import cv2
import numpy as np

dataset_files = {}
with open('lists/train.txt', 'r') as f:
    dataset_files['train'] = f.readlines()

with open('lists/test.txt', 'r') as f:
    dataset_files['test'] = f.readlines()

row = 200
col = 200

img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')

dataset = {}
labels = {}
for mode in ['train', 'test']:
    length = len(dataset_files[mode])
    dataset[mode] = np.zeros((length, row, col, 3))
    labels[mode] = np.zeros((length, 1), dtype=np.int32)

from tqdm import tqdm
for mode in ['train', 'test']:
    for i, filename in enumerate(dataset_files[mode]):
        im = cv2.resize(
            cv2.imread(os.path.join(img_dir, filename.strip())),
            (col, row)
        )
        idx = int(filename[:3]) - 1
        dataset[mode][i] = im
        labels[mode][i] = idx

for mode in ['train', 'test']:
    T.save({
        'data': T.FloatTensor(dataset[mode]),
        'labels': T.LongTensor(labels[mode])
    }, '{}.pt'.format(mode))
