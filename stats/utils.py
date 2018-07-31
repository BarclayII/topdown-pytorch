import torch as T
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def glimpse_to_xyhw(glim_params):
    if isinstance(glim_params, list):
        glim_params = np.array(glim_params)
    if T.is_tensor(glim_params):
        glim_params = glim_params.cpu().numpy()

    ret = np.zeros((glim_params.shape[0], 4))
    ret[:, 0] = glim_params[:, 0] - glim_params[:, 2] / 2.0
    ret[:, 1] = glim_params[:, 1] - glim_params[:, 3] / 2.0
    ret[:, 2] = glim_params[:, 2]
    ret[:, 3] = glim_params[:, 3]
    return ret

def to_numpy(func):
    def wrapper(*args, **kwargs):
        new_args = []
        for element in args:
            if isinstance(element, list):
                new_element = []
                for elem in element:
                    if isinstance(elem, list):
                        new_element.append(np.array(elem))
                    elif T.is_tensor(elem):
                        new_element.append(elem.cpu().numpy())
                    else:
                        new_element.append(elem)
            elif T.is_tensor(element):
                new_element = element.cpu().numpy()
            else:
                new_element = element

            new_args.append(new_element)
        return func(*new_args, **kwargs)
    return wrapper

def visualize_image_gray(fig, save_path=None):
    if not isinstance(fig, np.ndarray):
        fig = fig.numpy()
    plt.imshow(fig, 'cmap')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

class StatPlot(object):
    def __init__(self, canvas_x=1, canvas_y=1, figsize=None):
        self.canvas_x = canvas_x
        self.canvas_y = canvas_y
        if figsize is None:
            figsize = (canvas_y * 6, canvas_x * 4.5)
        self.fig, self.ax = plt.subplots(canvas_x, canvas_y, figsize=figsize)
        self.axs = self.ax_iter()

    def ax_iter(self):
        if isinstance(self.ax, np.ndarray):
            for ax in self.ax.flatten():
                yield ax
        else:
            yield self.ax

    def show(self):
        self.fig.show()

    def savefig(self, *args, **kwargs):
        self.fig.savefig(*args, **kwargs)

    @to_numpy
    def add_image(self, fig, x_label=None, y_label=None, title=None, bboxs=None, clrs=None, lws=None):
        """
        Args:
        lws: Linewidths
        """
        try:
            ax = next(self.axs)
        except StopIteration:
            print("Too many plots")
        else:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)

            assert fig.ndim == 3 or fig.ndim == 2
            if fig.ndim == 2:
                ax.imshow(fig, cmap='gray')
            else:
                ax.imshow(fig)

            if bboxs:
                if clrs is None:
                    clrs = ['r'] * len(bboxs)
                if lws is None:
                    lws = [1 * len(bboxs)]
                for bbox, clr, lw in zip(bboxs, clrs, lws):
                    x, y, h, w = bbox
                    rect = patches.Rectangle((x, y), h, w, linewidth=lw, edgecolor=clr, facecolor='none')
                    ax.add_patch(rect)

    @to_numpy
    def add_bar(self, xs, ys, x_label=None, y_label=None, title=None, labels=None, loc='upper right'):
        try:
            ax = next(self.axs)
        except StopIteration:
            print("Too many plots")
        else:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)

            if xs is None:
                xs = [xs] * len(ys)
            for x, y, label in zip(xs, ys, labels):
                if x is None:
                    x = np.arange(len(y))
                ax.bar(x, y, label=label)
            ax.legend(loc=loc)

    @to_numpy
    def add_curve(self, xs, ys, x_label=None, y_label=None, title=None, labels=None, loc='upper right'):
        try:
            ax = next(self.axs)
        except StopIteration:
            print("Too many plots")
        else:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)

            if xs is None:
                xs = [xs] * len(ys)
            for x, y, label in zip(xs, ys, labels):
                if x is None:
                    x = np.arange(len(y))
                ax.plot(x, y, label=label)
            ax.legend(loc=loc)

    @to_numpy
    def add_curve_with_mean_var(self, xs, means, stds, x_label=None, y_label=None, title=None, labels=None, loc='upper right'):
        try:
            ax = next(self.axs)
        except StopIteration:
            print("Too many plots")
        else:
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)

            if xs is None:
                xs = [xs] * len(stds)
            for x, mean, std, label in zip(xs, means, stds, labels):
                if x is None:
                    x = np.arange(len(mean))
                ax.plot(x, mean, label=label)
                ax.fill_between(x, mean - std, mean + std, alpha=0.3)
            ax.legend(loc=loc)
