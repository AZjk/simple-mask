import numpy as np
import h5py
import pyqtgraph as pg
from pyqtgraph import QtCore
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


def normalize(arr):
    # normalize arr so it's range is between 0 and 1
    vmin = np.min(arr)
    vmax = np.max(arr)

    if vmax - vmin > 1E-10:
        return (arr - vmin) / (vmax - vmin)
    else:
        return np.copy(arr)


class SimpleMask(object):
    def __init__(self, pg_hdl, infobar):
        self.data_raw = None
        self.det_dist = None
        self.pix_dim = None
        self.center = None
        self.energy = None
        self.shape = None
        self.qmap = None
        self.mask = None 
        self.is_rotate = False

        self.hdl = pg_hdl
        self.infobar = infobar
        self.extent = None
        self.hdl.scene.sigMouseMoved.connect(self.show_location)

        self.idx_map = {
            0: "scattering",
            # 1: "scattering * (1 - mask)",
            1: "scattering * mask",
            2: "mask",
            # 4: "qr",
            # 5: "qx",
            # 6: "qy",
            3: "dqmap_partition",
            4: "sqmap_partition"
        }

    def read_data(self, fname=None):
        with h5py.File(fname, 'r') as f:
            saxs = f[keymap['saxs_2d']][()]
            ccd_x0 = np.squeeze(f[keymap['ccd_x0']][()])
            ccd_y0 = np.squeeze(f[keymap['ccd_y0']][()])
            self.energy = np.squeeze(f[keymap['X_energy']][()])
            self.det_dist = np.squeeze(f[keymap['det_dist']][()])
            self.pix_dim = np.squeeze(f[keymap['pix_dim']][()])

        self.data_raw = np.zeros(shape=(5, *saxs.shape))
        self.mask = np.ones(saxs.shape, dtype=np.bool)

        self.center = (ccd_y0, ccd_x0)
        self.shape = self.data_raw.shape
        self.qmap = self.compute_qmap()

        self.extent = self.compute_extent()

        min_val = np.min(saxs[saxs > 0])
        saxs = np.log10(saxs + min_val) 
        self.data_raw[0] = saxs 

        # set the default values
        self.data_raw[1] = saxs * self.mask 
        self.data_raw[2] = self.mask 


    def compute_qmap(self):
        k0 = 2 * np.pi / self.energy
        v = np.arange(self.shape[1], dtype=np.uint32) - self.center[0]
        h = np.arange(self.shape[2], dtype=np.uint32) - self.center[1]
        vg, hg = np.meshgrid(v, h, indexing='ij')

        r = np.sqrt(vg * vg + hg * hg) * self.pix_dim
        phi = np.arctan2(vg, hg)
        phi[phi < 0] = phi[phi < 0] + np.pi * 2.0

        alpha = np.arctan(r / self.det_dist)
        qr = np.sin(alpha) * k0
        qx = qr * np.cos(phi)
        qy = qr * np.sin(phi)

        res = {
            'phi': phi.astype(np.float32),
            'alpha': alpha.astype(np.float32),
            'qr': qr.astype(np.float32),
            'qx': qx.astype(np.float32),
            'qy': qy.astype(np.float32)
        }
        return res

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

    def show_location(self, pos):

        if not self.hdl.scene.itemsBoundingRect().contains(pos) or \
           self.shape is None:
            return

        shape = self.shape[1:]
        mouse_point = self.hdl.getView().mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())

        if col < 0 or col >= shape[1]:
            return
        if row < 0 or row >= shape[0]:
            return

        qx = self.qmap['qx'][row, col]
        qy = self.qmap['qy'][row, col]
        phi = self.qmap['phi'][row, col] * 180 / np.pi
        val = self.data[self.hdl.currentIndex][row, col]

        # msg = f'{self.idx_map[self.hdl.currentIndex]}: ' + \
        msg = f'[x={col:4d}, y={row:4d}, ' + \
              f'qx={qx:.04f}Å⁻¹, qy={qy:.06f}Å⁻¹, phi={phi:.1f}deg], ' + \
              f'val={val}'

        self.infobar.clear()
        self.infobar.setText(msg)

        return None

    def show_saxs(self, cmap='jet', log=True, invert=False, rotate=False,
                  plot_center=True, plot_index=0, **kwargs):
        # self.hdl.reset_limits()
        self.hdl.clear()
        self.data = np.copy(self.data_raw)

        center = list(self.center).copy()
        if rotate:
            self.data = np.swapaxes(self.data, 1, 2)
            center = [center[1], center[0]]
        self.is_rotate = rotate
        
        if not log:
            self.data[0] = 10 ** self.data[0]
        
        if invert:
            temp = np.max(self.data[0]) - self.data[0]
            self.data[0] = temp

        self.hdl.setImage(self.data)
        self.hdl.adjust_viewbox()
        self.hdl.set_colormap(cmap)

        # plot center
        if plot_center:
            t = pg.ScatterPlotItem()
            t.addPoints(x=[center[1]], y=[center[0]], symbol='+', size=15)
            self.hdl.add_item(t)
        
        self.hdl.setCurrentIndex(plot_index)

        return

    def apply_roi(self):
        if len(self.hdl.roi) <= 0:
            return

        ones = np.ones(self.data[0].shape, dtype=np.bool)
        mask_n = np.zeros_like(ones, dtype=np.bool)
        mask_e = np.zeros_like(mask_n) 
        mask_i = np.zeros_like(mask_n) 

        for x in self.hdl.roi:
            # get ride of the center plot if it's there
            if isinstance(x, pg.ScatterPlotItem):
                continue
            # else
            mask_temp = np.zeros_like(ones, dtype=np.bool)
            # return slice and transfrom
            sl, _ = x.getArraySlice(self.data[1], self.hdl.imageItem)
            y = x.getArrayRegion(ones, self.hdl.imageItem)

            # sometimes the roi size returned from getArraySlice and 
            # getArrayRegion are different; 
            sl_v = slice(sl[0].start, sl[0].start + y.shape[0])
            sl_h = slice(sl[1].start, sl[1].start + y.shape[1])
            mask_temp[sl_v, sl_h] = y

            if x.sl_mode == 'exclusive':
                mask_e[mask_temp] = 1
            elif x.sl_mode == 'inclusive':
                mask_i[mask_temp] = 1

        if np.sum(mask_i) == 0:
            mask_i = 1

        mask_p = np.logical_not(mask_e) * mask_i

        self.mask = mask_p
        # self.data[1] = self.data[0] * (1 - mask_p)
        self.data[1] = self.data[0] * mask_p
        self.data[2] = self.mask 
        self.hdl.repaint()
        self.hdl.setCurrentIndex(2)

    def add_roi(self, num_edges=None, radius=60, color='r', sl_type='Polygon',
                width=3, sl_mode='exclusive'):

        shape = self.data.shape
        cen = (shape[1] // 2, shape[2] // 2)
        if sl_mode == 'inclusive':
            pen = pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
        else:
            pen = pg.mkPen(color=color, width=width)

        if sl_type == 'Ellipse':
            new_roi = pg.EllipseROI([cen[1], cen[0]], [60, 80], pen=pen, 
                                    removable=True, hoverPen=pen)
            # add scale handle
            new_roi.addScaleHandle([0.5, 0], [0.5, 1])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])            
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])
        
        elif sl_type == 'Circle':
            new_roi = pg.CircleROI([cen[1], cen[0]], [60, 80], pen=pen, 
                                   removable=True, hoverPen=pen)

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
                                     removable=True, hoverPen=pen)

        elif sl_type == 'Rectangle':
            new_roi = pg.RectROI([cen[1], cen[0]], [30, 150], pen=pen,
                                 removable=True, hoverPen=pen)
            new_roi.addScaleHandle([0, 0], [1, 1])
            new_roi.addRotateHandle([0,1], [0.5, 0.5])

        else:
            raise TypeError('type not implemented. %s' % sl_type)

        new_roi.sl_mode = sl_mode
        self.hdl.add_item(new_roi)
        new_roi.sigRemoveRequested.connect(lambda: self.remove_roi(new_roi))
        return
    
    def remove_roi(self, roi):
        self.hdl.remove_item(roi)

    def compute_partition(self, dq_num=10, sq_num=100, mode='linear',
                          dp_num=36, sp_num=360):
        if sq_num % dq_num != 0:
            raise ValueError('sq_num must be multiple of dq_num')

        if sp_num % dp_num != 0:
            raise ValueError('sq_num must be multiple of dq_num')

        qmap = self.qmap['qr']
        qmap_valid = qmap[self.mask == True]

        qmin = np.min(qmap_valid)
        qmax = np.max(qmap_valid)

        if mode == 'linear':
            dqlist = np.linspace(qmin, qmax, dq_num + 1)
            sqlist = np.linspace(qmin, qmax, sq_num + 1)
            dphi = np.linspace(0, np.pi * 2.0, dp_num + 1)
            sphi = np.linspace(0, np.pi * 2.0, sp_num + 1)

        dqmap_partition = np.zeros_like(qmap, dtype=np.uint32)
        sqmap_partition = np.zeros_like(qmap, dtype=np.uint32)

        for n in range(dq_num):
            qval = dqlist[n]
            dqmap_partition[qmap >= qval] = n + 1

        for n in range(sq_num):
            qval = sqlist[n]
            sqmap_partition[qmap >= qval] = n + 1

        dphi_partition = np.zeros_like(qmap, dtype=np.uint32)
        sphi_partition = np.zeros_like(qmap, dtype=np.uint32)

        for n in range(dp_num):
            dphi_partition[self.qmap['phi'] >= dphi[n]] = n

        for n in range(sp_num):
            sphi_partition[self.qmap['phi'] >= sphi[n]] = n

        dyn_combined = np.zeros_like(dqmap_partition, dtype=np.uint32)
        sta_combined = np.zeros_like(dqmap_partition, dtype=np.uint32)

        for n in range(dp_num):
            idx = dphi_partition == n
            dyn_combined[idx] = dqmap_partition[idx] + n * dq_num

        for n in range(sp_num):
            idx = sphi_partition == n
            sta_combined[idx] = sqmap_partition[idx] + n * sq_num

        self.data[3] = dyn_combined * self.mask
        self.data[4] = sta_combined * self.mask
        
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