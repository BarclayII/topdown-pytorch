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

def _place_glimpse_on_image(img, row, col, bbox, glim):
    x, y, h, w = bbox
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)
    if h <= 0 or w <= 0:
        return
    rescaled_glim = cv2.resize(
        glim,
        (h, w) #(w, h)
    )

    x_src = 0
    y_src = 0
    if x >= col or y >= row or x + h < 0 or y + w < 0:
        return

    if x + h >= col:
        h = col - x
    if y + w >= row:
        w = row - y
    if x < 0:
        x_src = -x
        x = 0
        h -= x_src
    if y < 0:
        y_src = -y
        y = 0
        w -= y_src
    img[y: y + w, x: x + h, :] = \
        rescaled_glim[y_src: y_src + w,x_src: x_src + h,  :]

def display_glimpse(channel, row, col, bbox_list, glim_list):
    ret_img = np.zeros((row, col, channel))
    size_idx_list = list(
        zip([h * w for (x, y, h, w) in bbox_list], range(len(bbox_list))))
    size_idx_list.sort(
        key = lambda x: x[0], reverse=True)
    idx_orderlist = [_[1] for _ in size_idx_list]
    for idx in idx_orderlist:
        bbox = bbox_list[idx]
        glim = glim_list[idx]
        if isinstance(glim, T.Tensor):
            glim = glim.cpu().numpy()
        _place_glimpse_on_image(ret_img, row, col, bbox, glim)
    return ret_img
