import torch as T
import numpy as np
import cv2

def resize_img(img, x, y):
    channel = img.shape[-1]
    ret_img = np.zeros((x, y, channel))
    for i in range(x):
        for j in range(y):
            i_src = int(i * img.shape[0] / x)
            j_src = int(j * img.shape[1] / y)
            ret_img[i, j, :] = img[i_src, j_src, :]
    return ret_img

def _place_glimpse_on_image(img, bbox, glim):
    x, y, w, h = bbox
    row, col = img.shape[:2]
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    if h <= 0 or w <= 0:
        return

    rescaled_glim = cv2.resize(
        glim,
        (w, h) #(h, w) #(h, w)
    )

    x_src = 0
    y_src = 0
    if x >= col or y >= row or x + w < 0 or y + h < 0:
        return

    if x + w >= col:
        w = col - x
    if y + h >= row:
        h = row - y
    if x < 0:
        x_src = -x
        x = 0
        w -= x_src
    if y < 0:
        y_src = -y
        y = 0
        h -= y_src

    img[y: y + h, x: x + w, :] = \
        rescaled_glim[y_src: y_src + h, x_src: x_src + w, :]

def display_glimpse(channel, row, col, bbox_list, glim_list):
    ret_img = np.zeros((row, col, channel))
    size_idx_list = list(
        zip([h * w for (x, y, w, h) in bbox_list], range(len(bbox_list))))
    size_idx_list.sort(
        key = lambda x: x[0], reverse=True)
    idx_orderlist = [_[1] for _ in size_idx_list]
    for idx in idx_orderlist:
        bbox = bbox_list[idx]
        glim = glim_list[idx]
        if isinstance(glim, T.Tensor):
            glim = glim.cpu().numpy()
        glim = glim
        _place_glimpse_on_image(ret_img, bbox, glim)
    return ret_img
