import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
import warnings

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path


# hdf keys for APS 8idi data format
with open('hdf_config.json', 'r') as f:
    keymap = json.load(f)


class SimpleMask(object):
    def __init__(self):
        self.saxs = None
        self.det_dist = None
        self.pix_dim = None
        self.center = None
        self.energy = None
        self.history = []
        self.curr_ptr = 0
        self.shape = None
        self.vh = None
        self.vhq = None

    def read_data(self, fname):
        with h5py.File(fname, 'r') as f:
            saxs = f[keymap['saxs_2d']][()]
            ccd_x0 = np.squeeze(f[keymap['ccd_x0']][()])
            ccd_y0 = np.squeeze(f[keymap['ccd_y0']][()])
            self.energy = np.squeeze(f[keymap['X_energy']][()])
            self.det_dist = np.squeeze(f[keymap['det_dist']][()])
            self.pix_dim = np.squeeze(f[keymap['pix_dim']][()])

        # find min vaule to compute log
        min_val = np.min(saxs[saxs > 0])
        self.saxs = np.log10(saxs + min_val)
        self.center = (ccd_y0, ccd_x0)
        self.shape = self.saxs.shape
        self.vh, self.vhq = self.compute_map()

    def compute_map(self):
        k0 = 2 * np.pi / self.energy
        v = np.arange(self.shape[0], dtype=np.uint32)
        h = np.arange(self.shape[1], dtype=np.uint32)
        vg, hg = np.meshgrid(v, h, indexing='ij')
        vq = (vg - self.center[0]) * self.pix_dim / self.det_dist * k0
        hq = (hg - self.center[1]) * self.pix_dim / self.det_dist * k0

        vh = np.vstack([vg.ravel(), hg.ravel()]).T
        print(np.max(vh[:, 0]), np.max(vh[:, 1]))
        return vh, (vq, hq)

    def compute_extent(self):
        k0 = 2 * np.pi / self.energy
        x_range = np.array([0, self.shape[1]]) - self.center[1]
        y_range = np.array([-self.shape[0], 0]) + self.center[0]
        x_range = x_range * self.pix_dim / self.det_dist * k0
        y_range = y_range * self.pix_dim / self.det_dist * k0
        # the extent for matplotlib imshow is:
        # self._extent = xmin, xmax, ymin, ymax = extent
        # convert to a tuple of 4 elements;
        return (*x_range, *y_range)

    def show_saxs(self):
        extent = self.compute_extent()
        plt.imshow(self.saxs, extent=extent)
        plt.show()

    def draw_roi(self):
        fig, [ax0, ax1] = plt.subplots(1, 2, sharex=True, sharey=True)
        self.canvas = fig.canvas
        self.ax1 = ax1
        ax0.imshow(self.saxs)
        ax1.imshow(self.get_mask())
        self.poly = PolygonSelector(ax0, self.onselect)
        plt.show()

    def onselect(self, verts):
        path = Path(verts)
        # contains_points take (x, y) list
        xys = np.roll(self.vh, 1, axis=1)
        ind = np.nonzero(path.contains_points(xys))[0]

        ind = self.vh[ind]
        ind = (ind[:, 0], ind[:, 1])
        mask = self.get_mask()
        mask[ind] = 0

        if self.curr_ptr == len(self.history):
            self.history.append(mask)
        else:
            self.history[self.curr_ptr] = mask

        self.curr_ptr += 1

        while self.curr_ptr < len(self.history):
            self.history.pop(self.curr_ptr)

        self.ax1.imshow(self.get_mask())
        self.canvas.draw_idle()
        # self.show_mask()

    def redo(self):
        pass

    def undo(self):
        if len(self.history) == 0:
            warnings.warn('nothing has been done')
            return
        if self.curr_ptr <= 0:
            warnings.warn('reach the initial point')
            return

        self.curr_ptr -= 1

        return

    def get_mask(self, ptr=None):
        if ptr is None:
            ptr = self.curr_ptr
        if ptr == 0:
            return np.ones(self.shape, dtype=np.uint32)
        else:
            return self.history[ptr - 1]

    def show_mask(self):
        plt.imshow(self.get_mask())
        plt.show()


def test01():
    fname = '../data/H187_D100_att0_Rq0_00001_0001-100000.hdf'
    sm = SimpleMask()
    sm.read_data(fname)
    # sm.show_saxs()
    # sm.compute_qmap()
    sm.draw_roi()


if __name__ == '__main__':
    test01()