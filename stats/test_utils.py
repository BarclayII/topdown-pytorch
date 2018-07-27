from utils import StatPlot
from viz import VisdomWindowManager

statplot = StatPlot(2, 3)
statplot.add_curve(None, [[2, 3, 4, 1, 2, 3, 5], [3, 3, 2]], labels=['curve_1', 'curve_2'])
statplot.add_curve_with_mean_var(None, [[4, 5, 3, 6, 2, 5, 1]], [[0.2] * 7], labels=['excellent'], x_label='step', y_label='val', title='test')
import numpy as np
statplot.add_curve(None, [np.random.rand(100)], labels=['random'])
statplot.add_image(np.random.rand(100, 100), bboxs=[(20, 20, 20, 20)])
statplot.add_image(np.random.rand(80, 80), bboxs=[(10, 20, 5, 15), (10, 13, 10, 10)], clrs=['r', 'g'])
wm = VisdomWindowManager(port=11111)
wm.append_scalar('loss', 121.5)
wm.append_scalar('loss', 101.2)
#statplot.savefig('try')
