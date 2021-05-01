from PyQt5 import QtCore
from simple_mask_ui import Ui_MainWindow as Ui
from simple_mask_kernel import SimpleMask
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import pyqtgraph as pg
from matplotlib.backend_bases import KeyEvent

import os
import numpy as np
import sys
import json
import shutil
import logging


home_dir = os.path.join(os.path.expanduser('~'), '.simple-mask')
if not os.path.isdir(home_dir):
    os.mkdir(home_dir)
log_filename = os.path.join(home_dir, 'viewer.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-24s: %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename, mode='a'),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)


def exception_hook(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception",
                 exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = exception_hook


class SimpleMaskGUI(QtWidgets.QMainWindow, Ui):
    def __init__(self, path=None):
        super(SimpleMaskGUI, self).__init__()
        self.setupUi(self)

        # more setup; buttons
        self.btn_load.clicked.connect(self.load)
        self.btn_add_roi.clicked.connect(self.add_roi)
        self.btn_apply_roi.clicked.connect(self.apply_roi)
        self.btn_plot.clicked.connect(self.plot)
        self.btn_editlock.clicked.connect(self.editlock)
        self.btn_compute_qpartition.clicked.connect(self.compute_partition)

        self.plot_index.currentIndexChanged.connect(self.mp1.setCurrentIndex)

        # simple mask kernel
        self.sm = SimpleMask(self.mp1, self.infobar)
        self.mp1.sigTimeChanged.connect(self.update_index)

        self.state = 'lock'
        self.show()

    def update_index(self):
        idx = self.mp1.currentIndex
        self.plot_index.setCurrentIndex(idx)

    def editlock(self):
        pvs = (self.db_cenx, self.db_ceny, self.db_energy, self.db_pix_dim,
               self.db_det_dist)

        if self.state == 'lock':
            self.state = 'edit'
            for pv in pvs:
                pv.setEnabled(True)
        elif self.state == 'edit':
            self.state = 'lock'
            values = []
            for pv in pvs:
                pv.setDisabled(True)
                values.append(pv.value())

            # update value
            self.sm.update_parameters(values)

        self.groupBox.repaint()
        self.plot()

    def load(self):
        fname = QFileDialog.getOpenFileName(self, 'Open directory')[0]
        # fname = "/Users/mqichu/Documents/local_dev/xpcs_mask/data/H187_D100_att0_Rq0_00001_0001-100000.hdf"
        self.fname.setText(os.path.basename(fname))
        self.sm.read_data(fname)

        self.db_cenx.setValue(self.sm.center[1])
        self.db_ceny.setValue(self.sm.center[0])
        self.db_energy.setValue(self.sm.energy)
        self.db_pix_dim.setValue(self.sm.pix_dim)
        self.db_det_dist.setValue(self.sm.det_dist)
        self.le_shape.setText(str(self.sm.shape[1:]))
        self.groupBox.repaint()
        self.plot()

    def plot(self):
        kwargs = {
            'cmap': self.plot_cmap.currentText(),
            'log': self.plot_log.isChecked(),
            'invert': self.plot_invert.isChecked(),
            'rotate': self.plot_rotate.isChecked(),
            'plot_center': self.plot_center.isChecked(),
        }
        self.sm.show_saxs(**kwargs)
        self.plot_index.setCurrentIndex(0)

    def add_roi(self):
        color = ('g', 'y', 'b', 'r', 'c', 'm', 'k', 'w')[
                self.cb_selector_color.currentIndex()]
        kwargs = {
            'color': color,
            'sl_type': self.cb_selector_type.currentText(),
            'sl_mode': self.cb_selector_mode.currentText(),
            'width': self.plot_width.value()
        }
        self.sm.add_roi(**kwargs)
        return

    def apply_roi(self):
        self.sm.apply_roi()
        self.plot_index.setCurrentIndex(2)
        return 
    
    def compute_partition(self):
        kwargs = {
            'sq_num': self.sb_sqnum.value(),
            'dq_num': self.sb_dqnum.value(),
            'sp_num': self.sb_spnum.value(),
            'dp_num': self.sb_dpnum.value(),
        }
        self.sm.compute_partition(**kwargs)
        self.plot_index.setCurrentIndex(3)


def run():
    # if os.name == 'nt':
    #     setup_windows_icon()
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    window = SimpleMaskGUI()
    app.exec_()


if __name__ == '__main__':
    run()
