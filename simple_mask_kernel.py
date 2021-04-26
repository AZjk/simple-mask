import numpy as np
import h5py
import pyqtgraph as pg
import json
import warnings
from matplotlib import cm

from matplotlib.widgets import (
    PolygonSelector, EllipseSelector, RectangleSelector, LassoSelector)
from matplotlib.path import Path

pg.setConfigOptions(imageAxisOrder='row-major')


# hdf keys for APS 8idi data format
with open('hdf_config.json', 'r') as f:
    keymap = json.load(f)


class SimpleMask(object):
    def __init__(self, pg_hdl):
        self.saxs = None
        self.det_dist = None
        self.pix_dim = None
        self.center = None
        self.energy = None
        self.shape = None
        self.vh = None
        self.vhq = None
        self.sl_type = None

        self.hdl = pg_hdl
        self.extent = None

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
        saxs = np.log10(saxs + min_val)
        # normalize saxs
        vmin = np.min(saxs)
        vmax = np.max(saxs)
        saxs = (saxs - vmin) / (vmax - vmin)
        self.saxs = np.zeros(shape=(5, *saxs.shape))
        self.saxs[0] = saxs
        self.saxs[-1][:, :] = 1
        self.saxs_raw = np.copy(self.saxs)

        self.center = (ccd_y0, ccd_x0)
        self.shape = self.saxs.shape
        self.vh, self.vhq = self.compute_map()

        self.extent = self.compute_extent()

    def compute_map(self):
        k0 = 2 * np.pi / self.energy
        v = np.arange(self.shape[0], dtype=np.uint32)
        h = np.arange(self.shape[1], dtype=np.uint32)
        vg, hg = np.meshgrid(v, h, indexing='ij')
        vq = (vg - self.center[0]) * self.pix_dim / self.det_dist * k0
        hq = (hg - self.center[1]) * self.pix_dim / self.det_dist * k0

        vh = np.vstack([vg.ravel(), hg.ravel()]).T
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

    def show_location(self, event):
        if event.xdata is None or event.ydata is None:
            return None

        et = self.extent
        shape = self.shape
        if 0 <= event.xdata < shape[1]:
            if 0 <= event.ydata < shape[0]:
                kx = event.xdata * (et[1] - et[0]) / shape[1] + et[0]
                ky = event.ydata * (et[3] - et[2]) / shape[0] + et[2]
                kxy = np.sqrt(kx * kx + ky * ky)
                phi = np.rad2deg(np.arctan2(ky, kx))
                if phi < 0:
                    phi += 360
                return f'kx={kx:.5f}Å⁻¹, ky={ky:.5f}Å⁻¹, kxy={kxy:.5f}Å⁻¹, '\
                       f'phi={phi:.1f}deg'
        return None

    def show_saxs(self, cmap='jet', log=True, invert=False, rotate=False,
                  plot_center=True, plot_index=0, **kwargs):
        self.hdl.clear()
        self.saxs = np.copy(self.saxs_raw)

        center = list(self.center).copy()
        if rotate:
            self.saxs = np.swapaxes(self.saxs, 1, 2)
            center = [center[1], center[0]]
        
        if not log:
            self.saxs[0] = 10 ** self.saxs[0]
        
        if invert:
            temp = np.max(self.saxs[0]) - self.saxs[0]
            self.saxs[0] = temp

        self.hdl.setImage(self.saxs)
        self.hdl.adjust_viewbox()
        self.hdl.set_colormap(cmap)

        # plot center
        if plot_center:
            t = pg.ScatterPlotItem()
            t.addPoints(x=[center[1]], y=[center[0]], symbol='+')
            self.hdl.add_item(t)
        
        self.hdl.setCurrentIndex(plot_index)

        return

    def apply_roi(self):
        if len(self.hdl.roi) <= 0:
            return

        ones = np.ones(self.saxs[0].shape, dtype=np.bool)
        mask_n = np.zeros_like(ones, dtype=np.bool)

        for x in self.hdl.roi:
            # get ride of the center plot if it's there
            if isinstance(x, pg.ScatterPlotItem):
                continue
            # else
            mask_n_temp = np.zeros_like(ones, dtype=np.bool)
            # return slice and transfrom
            sl, _ = x.getArraySlice(self.saxs[1], self.hdl.imageItem)
            y = x.getArrayRegion(ones, self.hdl.imageItem)

            # sometimes the roi size returned from getArraySlice and 
            # getArrayRegion are different; 
            sl_v = slice(sl[0].start, sl[0].start + y.shape[0])
            sl_h = slice(sl[1].start, sl[1].start + y.shape[1])
            mask_n_temp[sl_v, sl_h] = y
            mask_n = np.logical_or(mask_n, mask_n_temp)
        
        mask_p = np.logical_not(mask_n)
        self.saxs[1] = self.saxs[0] * mask_n
        self.saxs[2] = self.saxs[0] * mask_p
        self.saxs[3] = 1 * mask_p
        self.hdl.repaint()
        self.hdl.setCurrentIndex(1)

    def add_roi(self, num_edges=None, radius=60, color='r', sl_type='Polygon',
                width=3):
        shape = self.saxs.shape
        cen = (shape[1] // 2, shape[2] // 2)
        pen = pg.mkPen(color=color, width=width)

        if sl_type == 'Ellipse':
            new_roi = pg.EllipseROI([cen[1], cen[0]], [60, 80], pen=pen, 
                                    removable=True)
            # add scale handle
            new_roi.addScaleHandle([0.5, 0], [0.5, 1])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])            
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])

        elif sl_type == 'Polygon':
            if num_edges is None:
                num_edges = np.random.random_integers(6, 10)

            # add angle offset so that the new rois don't overlap with each other
            offset = np.random.random_integers(0, 359)
            theta = np.linspace(0, np.pi * 2, num_edges + 1) + offset
            x = radius * np.cos(theta) + cen[1]
            y = radius * np.sin(theta) + cen[0]
            pts = np.vstack([x, y]).T
            new_roi = pg.PolyLineROI(pts, closed=True, pen=pen,
                                     removable=True)

        elif sl_type == 'Rectangle':
            new_roi = pg.RectROI([cen[1], cen[0]], [30, 150], pen=pen,
                                 removable=True)
            new_roi.addScaleHandle([0, 0], [1, 1])
            new_roi.addRotateHandle([0,1], [0.5, 0.5])

        else:
            raise TypeError('type not implemented. %s' % sl_type)

        self.hdl.add_item(new_roi)
        new_roi.sigRemoveRequested.connect(lambda: self.remove_roi(new_roi))
        return
    
    def remove_roi(self, roi):
        self.hdl.remove_item(roi)

    def compute_qmap(self, dq_num: int, sq_num: int, mode='linear'):
        if sq_num % dq_num != 0:
            raise ValueError('sq_num must be multiple of dq_num')

        mask = self.get_mask()
        qmap = np.sqrt(self.vhq[0] ** 2 + self.vhq[1] ** 2)
        qmap = qmap[mask == 1]

        qmin = np.min(qmap)
        qmax = np.max(qmap)

        if mode == 'linear':
            qlist = np.linspace(qmin, qmax, dq_num + 1)

        qindex = np.zeros(shape=self.shape, dtype=np.uint32)
        for n in range(dq_num):
            qval = qlist[n + 1]
            qindex[qmap ]

    def update_parameters(self, val):
        assert(len(val) == 5)
        self.center = (val[1], val[0])
        self.energy = val[2]
        self.pix_dim = val[3]
        self.det_dist = val[4]
    
    def get_parameters(self):
        val = (self.center[1], self.center[0], self.energy, self.pix_dim,
               self.det_dist)
        return val

def test01():
    fname = '../data/H187_D100_att0_Rq0_00001_0001-100000.hdf'
    sm = SimpleMask()
    sm.read_data(fname)
    # sm.show_saxs()
    # sm.compute_qmap()


if __name__ == '__main__':
    test01()