from .utils import StatPlot, to_numpy
import bisect

class NearestNeighborImageSet(object):
    @to_numpy
    def __init__(self, xbase, hbase, k=5, bboxs=None, clrs=None):
        self.k = k
        self.n = xbase.shape[0]
        self.i = 0

        self.q = [[] for _ in range(self.n)]
        self.xbase = xbase
        self.hbase = hbase.reshape(hbase.shape[0], -1)
        self.bboxs = bboxs
        self.clrs = clrs
        self.idx = 0

        self.stat_plot = StatPlot(self.n, k + 1)

    @to_numpy
    def push(self, x, h, bboxs=None, clrs=None):
        h = h.reshape(h.shape[0], -1)
        # dist[i, j] is the euclidean distance between h[j] and hbase[i]
        dist = ((h[None, :, :] - self.hbase[:, None, :]) ** 2).sum(2) ** 0.5

        for i in range(self.n):
            for j in range(h.shape[0]):
                bbox_i = [b[i] for b in bboxs] if bboxs is not None else None
                bisect.insort(self.q[i], (dist[i, j], self.idx, x[i], h[i], bbox_i, clrs))
                if len(self.q[i]) > self.k:
                    self.q[i].pop()
                self.idx += 1       # tie breaker

    def display(self):
        for i in range(self.n):
            bbox_i = [b[i] for b in self.bboxs] if self.bboxs is not None else None
            self.stat_plot.add_image(self.xbase[i], title='input image', bboxs=bbox_i, clrs=self.clrs)
            for item in sorted(self.q[i], key=lambda x: x[0]):
                d, j, x, h, b, c = item
                self.stat_plot.add_image(x, title='%.3f' % d, bboxs=b, clrs=c)
