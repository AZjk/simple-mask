import numpy as np
import time
import logging
import os
from tqdm import trange, tqdm
from xpcs_functions import convert_sparse, read_sparse_data
import glob2
import h5py
import json


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-24s: %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


class RigakuDataset(object):
    def __init__(self, folder, det_size=(512, 1024)):
        if os.path.isfile(folder):
            dir_name = os.path.dirname(folder)
        else:
            dir_name = folder
        
        bin_fname = glob2.glob(dir_name + '/*.bin')[0]
        hdf_fname = glob2.glob(dir_name + '/*.hdf')[0]

        csr, frame_num = read_sparse_data(bin_fname, 
                                          method='numpy',
                                          det_size=det_size)
        csr = np.sum(csr, axis=0)
        csr = csr.reshape(det_size)
        self.center, self.energy, self.det_dist, self.pix_dim = \
            self.read_data(hdf_fname)
        
        print(self.center)
    
    def read_data(self, fname=None):
        # hdf keys for APS 8idi data format
        with open('hdf_config.json', 'r') as f:
            keymap = json.load(f)

        with h5py.File(fname, 'r') as f:
            ccd_x0 = np.squeeze(f[keymap['ccd_x0']][()])
            ccd_y0 = np.squeeze(f[keymap['ccd_y0']][()])
            energy = np.squeeze(f[keymap['X_energy']][()])
            det_dist = np.squeeze(f[keymap['det_dist']][()])
            pix_dim = np.squeeze(f[keymap['pix_dim']][()])

        return (ccd_x0, ccd_y0), energy, det_dist, pix_dim





if __name__ == '__main__':
    RigakuDataset("/Users/mqichu/Documents/local_dev/xpcs_data/O018_Silica_D100_att0_Rq0_00005/")