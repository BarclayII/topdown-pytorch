import matplotlib.pyplot as plt
from util import *
import numpy as np
from stats import *

def fig_to_ndarray_tb(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


available_clrs = ['y', 'r', 'g', 'b']

def getclrs(n_branches, n_levels):
    clrs = []
    for j in range(n_levels):
        clrs += [available_clrs[j]] * (n_branches ** (j + 1))

    return clrs


def viz(epoch, imgs, bboxes, g_arr, alpha_arr, recon_arr, tag, writer, y_pred, y, n_branches=2, n_levels=2):
    length = len(g_arr)
    statplot = StatPlot(5, 2)
    statplot_g_arr = [StatPlot(5, 2) for _ in range(length)]
    statplot_alpha_arr = [StatPlot(5, 2) for _ in range(length)]
    statplot_recon_arr = [StatPlot(5, 2) for _ in range(length)]

    clrs = getclrs(n_branches, n_levels)
    for j in range(10):
        statplot.add_image(
            imgs[j].permute(1, 2, 0),
            bboxs=[bbox[j] for bbox in bboxes],
            clrs=clrs, #['y', 'y', 'r', 'r', 'r', 'r'],
            lws=[5] * length, #att[j, 1:] * length
            title='%d/%d' % (y_pred[j], y[j]),
        )
        for k in range(length):
            statplot_g_arr[k].add_image(g_arr[k][j].permute(1, 2, 0))
            statplot_alpha_arr[k].add_image(alpha_arr[k][j, 0])
            statplot_recon_arr[k].add_image(recon_arr[k][j].permute(1, 2, 0))

    statplot_disp_g = StatPlot(5, 2)
    channel, row, col = imgs[-1].shape
    for j in range(10):
        bbox_list = [
            np.array([0, 0, col, row])
        ] + [
            bbox_batch[j] for bbox_batch in bboxes
        ]
        glim_list = [
            g_arr[k][j].permute(1, 2, 0) for k in range(length)]
        statplot_disp_g.add_image(
            display_glimpse(channel, row, col, bbox_list, glim_list))
    writer.add_image('Image/{}/disp_glim'.format(tag), fig_to_ndarray_tb(statplot_disp_g.fig), epoch)
    writer.add_image('Image/{}/viz_bbox'.format(tag), fig_to_ndarray_tb(statplot.fig), epoch)
    for k in range(length):
        writer.add_image('Image/{}/viz_glim_{}'.format(tag, k), fig_to_ndarray_tb(statplot_g_arr[k].fig), epoch)
        writer.add_image('Image/{}/viz_alpha_{}'.format(tag, k), fig_to_ndarray_tb(statplot_alpha_arr[k].fig), epoch)
        writer.add_image('Image/{}/viz_recon_{}'.format(tag, k), fig_to_ndarray_tb(statplot_recon_arr[k].fig), epoch)
    plt.close('all')
