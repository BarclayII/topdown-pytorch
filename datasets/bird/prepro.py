import torch as T
import scipy.io
import os
import cv2
import numpy as np

filename_list = []

with open('CUB_200_2011/images.txt', 'r') as f:
    for i, line in enumerate(f):
        id, filename = line.strip().split()
        assert int(id) == i + 1
        filename_list.append(filename)

lbl_list = []

with open('CUB_200_2011/image_class_labels.txt', 'r') as f:
    for i, line in enumerate(f):
        id, lbl = line.strip().split()
        assert int(id) == i + 1
        lbl_list.append(int(lbl) - 1)

dataset_files = {
    'train': [],
    'test': []
}

with open('CUB_200_2011/train_test_split.txt', 'r') as f:
    for i, line in enumerate(f):
        id, is_train = line.strip().split()
        assert int(id) == i + 1
        if int(is_train) == 1:
            dataset_files['train'].append((filename_list[i], lbl_list[i]))
        else:
            dataset_files['test'].append((filename_list[i], lbl_list[i]))

row = 512 
col = 512

img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CUB_200_2011/images')
from tqdm import tqdm

for mode in ['train', 'test']:
    length = len(dataset_files[mode])
    dataset = np.zeros((length, row, col, 3))
    labels = np.zeros((length, 1), dtype=np.int32)

    for i, (filename, lbl) in tqdm(enumerate(dataset_files[mode])):
        im = cv2.resize(
            cv2.imread(os.path.join(img_dir, filename.strip())),
            (col, row)
        )
        idx = int(filename[:3]) - 1
        dataset[i] = im
        labels[i] = lbl

    T.save({
        'data': T.FloatTensor(dataset),
        'labels': T.LongTensor(labels)
    }, '{}.pt'.format(mode))
