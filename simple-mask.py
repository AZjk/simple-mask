import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
import warnings

with open('hdf_config.json', 'r') as f:
    keymap = json.load(f)


class SimpleMask(object):

    def __init__(self):
        self.saxs = None
        self.det_dist = None
        self.ccd_x0 = None
        self.ccd_y0 = None
        self.energy = None
        self.history = []
        self.curr_ptr = 0

    def read_data(self, fname):
        with h5py.File(fname, 'r') as f:
            saxs = f[keymap['saxs_2d']][()]
            ccd_x0 = np.squeeze(f[keymap['ccd_x0']][()])
            ccd_y0 = np.squeeze(f[keymap['ccd_y0']][()])

        # find min vaule to compute log
        min_val = np.min(saxs[saxs > 0])
        self.saxs = np.log10(saxs + min_val)

    def redo(self):
        pass

    def undo(self):
        if len(self.history) == 0:
            warnings.warn('nothing has been done')
            return
        if self.curr_ptr == 0:




def test01():
    fname = '../data/H187_D100_att0_Rq0_00001_0001-100000.hdf'
    sm = SimpleMask()
    sm.read_data(fname)


if __name__ == '__main__':
    test01()