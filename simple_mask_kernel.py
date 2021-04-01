import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
import json
import warnings

from matplotlib.widgets import (
    PolygonSelector, EllipseSelector, RectangleSelector, LassoSelector)
from matplotlib.path import Path


# hdf keys for APS 8idi data format
with open('hdf_config.json', 'r') as f:
    keymap = json.load(f)


class SimpleMask(object):
    def __init__(self, canvas=None, ax0=None, ax1=None):
        self.saxs = None
        self.det_dist = None
        self.pix_dim = None
        self.center = None
        self.energy = None
        self.history = []
        self.curr_ptr = -1
        self.shape = None
        self.vh = None
        self.vhq = None
        self.selector = None
        self.sl_type = None
        self.sl_color = None
        self.sl_cache = []

        # plot setting
        if canvas is None:
            fig, [ax0, ax1] = plt.subplots(1, 2, sharex=True, sharey=True)
            canvas = fig.canvas
        self.canvas = canvas
        self.ax0, self.ax1 = ax0, ax1
        self.xbound = None
        self.ybound = None

    def read_data(self, fname=None):
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

        self.history.append(np.ones(self.shape, dtype=np.uint32))
        self.curr_ptr = 0

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

    def draw_roi(self, canvas=None, ax0=None, ax1=None, invert=False,
                 log=True, vmin=0, vmax=100, cmap='jet'):
        mask = self.get_mask()

        if not log:
            data = 10 ** self.saxs
        else:
            data = self.saxs
        if invert:
            data = np.max(data) - data

        xmin, xmax = np.min(data), np.max(data)
        vmin = xmin + (xmax - xmin) * vmin / 100.0
        vmax = xmin + (xmax - xmin) * vmax / 100.0
        # data[mask == 0] = vmax

        self.xbound = self.ax0.get_xbound()
        self.ybound = self.ax0.get_ybound()

        self.ax0.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        self.ax0.plot(self.center[1], self.center[0], 'x', color='w', ms=2)
        self.ax1.imshow(self.get_mask(), vmin=0, vmax=1)

        if self.xbound != (0.0, 1.0):
            self.ax0.set_xbound(self.xbound)
            self.ax0.set_ybound(self.ybound)

    def select(self, sl_type='Polygon', color='y'):
        lineprops = dict(color=color, linestyle='-', linewidth=1, alpha=0.9)
        rectprops = dict(facecolor=color, edgecolor=color, alpha=0.9,
                         fill=True)

        if self.selector is not None:
            print('selector is not empty')
            return
        else:
            self.sl_type = sl_type
        self.sl_color = color

        if sl_type == 'Ellipse':
            self.selector = EllipseSelector(self.ax0, self.onselect,
                                            rectprops=rectprops)
        elif sl_type == 'Polygon':
            markerprops = dict(marker='s', markersize=3, mec=color, mfc=color,
                               alpha=0.9)
            self.selector = PolygonSelector(self.ax0, self.onselect,
                                            markerprops=markerprops,
                                            lineprops=lineprops)
        elif sl_type == 'Lasso':
            self.selector = LassoSelector(self.ax0, self.onselect,
                                          lineprops=lineprops)
        elif sl_type == 'Rectangle':
            self.selector = RectangleSelector(self.ax0, self.onselect,
                                              rectprops=rectprops)
        else:
            raise TypeError('type not implemented. %s' % sl_type)

    def onselect(self, *args):
        if len(args) > 2:
            raise ValueError('length of input > 2')
        # rectangle or ellipse selector;
        if len(args) == 2:
            x = [t.xdata for t in args]
            y = [t.ydata for t in args]
            x_cen = (x[0] + x[1]) / 2.0
            y_cen = (y[0] + y[1]) / 2.0
            x_rad = abs(x_cen - x[0])
            y_rad = abs(y_cen - y[0])

            if self.sl_type == 'Ellipse':
                phi = np.linspace(0, np.pi * 2, 256)
                x_arr = np.cos(phi) * x_rad + x_cen
                y_arr = np.sin(phi) * y_rad + y_cen
                verts = np.vstack([x_arr, y_arr]).T

            elif self.sl_type == 'Rectangle':
                verts = [
                    (x_cen - x_rad, y_cen - y_rad),
                    (x_cen + x_rad, y_cen - y_rad),
                    (x_cen + x_rad, y_cen + y_rad),
                    (x_cen - x_rad, y_cen + y_rad),
                ]
            else:
                raise TypeError('selector type not supported')
        # polygon or lasso; already a list of coordinates
        else:
            verts = args[0]

        path = Path(verts)
        patch = PathPatch(path, color=self.sl_color, edgecolor=None)
        self.ax0.add_patch(patch)
        mask = self.get_mask()
        new_mask = self.create_mask(mask, path)

        self.curr_ptr += 1
        if self.curr_ptr == len(self.history):
            self.history.append(new_mask)
        else:
            self.history[self.curr_ptr] = new_mask

        # remove all cache beyond this point
        for n in range(self.curr_ptr + 1, len(self.history)):
            self.history.pop(self.curr_ptr + 1)
        self.sl_cache = []

        self.draw_roi()
        return

    def create_mask(self, mask, path):
        # contains_points take (x, y) list
        xys = np.roll(self.vh, 1, axis=1)

        ind = np.nonzero(path.contains_points(xys))[0]
        ind = self.vh[ind]
        ind = (ind[:, 0], ind[:, 1])
        mask[ind] = 0

        return mask

    def finish(self, event):
        self.selector.on_key_press(event)
        self.selector.disconnect_events()
        self.selector = None
        self.canvas.draw_idle()

    def redo(self):
        if self.curr_ptr < len(self.history) - 1:
            self.curr_ptr += 1
            if len(self.sl_cache) > 0:
                t = self.sl_cache.pop(-1)
                self.ax0.add_patch(t)
            self.draw_roi()
        else:
            warnings.warn('at the end')

    def undo(self):
        if self.curr_ptr <= 0:
            warnings.warn('nothing has been done')
            return

        self.curr_ptr -= 1
        t = self.ax0.patches.pop(-1)
        self.sl_cache.append(t)
        self.draw_roi()

        return

    def get_mask(self, ptr=None):
        if ptr is None:
            ptr = self.curr_ptr
        if ptr == -1:
            # return a new full mask
            return np.ones(self.shape, dtype=np.uint32)
        else:
            # pass a copy instead of reference
            return np.copy(self.history[ptr])

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