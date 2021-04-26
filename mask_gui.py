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


format = logging.Formatter('%(asctime)s %(message)s')
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
        self.more_setup()
        self.show()
        self.cid = None
        self.state = 'lock'

    def more_setup(self):
        self.btn_load.clicked.connect(self.load)
        # self.btn_select.clicked.connect(self.select)

        self.btn_add_roi.clicked.connect(self.add_roi)
        self.btn_apply_roi.clicked.connect(self.apply_roi)

        self.btn_plot.clicked.connect(self.plot)
        self.btn_editlock.clicked.connect(self.editlock)

        self.sm = SimpleMask(self.mp1, None)

    def editlock(self):
        pvs = (self.db_cenx, self.db_ceny, self.db_energy, self.db_pix_dim,
               self.db_det_dist)

        if self.state == 'lock':
            self.state = 'edit'
            for pv in pvs:
                pv.setEnabled(True)
        elif self.state == 'edit':
            self.state = 'lock'
            for pv in pvs:
                pv.setDisabled(True)
        self.groupBox.repaint()

    def load(self):
        # fname = QFileDialog.getOpenFileName(self, 'Open directory')[0]
        fname = "/Users/mqichu/Documents/local_dev/xpcs_mask/data/H187_D100_att0_Rq0_00001_0001-100000.hdf"
        self.fname.setText(os.path.basename(fname))
        self.sm.read_data(fname)

        self.db_cenx.setValue(self.sm.center[1])
        self.db_ceny.setValue(self.sm.center[0])
        self.db_energy.setValue(self.sm.energy)
        self.db_pix_dim.setValue(self.sm.pix_dim)
        self.db_det_dist.setValue(self.sm.det_dist)
        self.le_shape.setText(str(self.sm.saxs[0].shape))
        self.groupBox.repaint()
        self.plot()

    def plot(self):
        kwargs = {
            'cmap': self.plot_cmap.currentText(),
            'log': self.plot_log.isChecked(),
            'invert': self.plot_invert.isChecked(),
            'rotate': self.plot_rotate.isChecked(),
        }
        self.sm.show_saxs(**kwargs)

    def add_roi(self):
        color = ('y', 'b', 'g', 'r', 'c', 'm', 'k', 'w')[
                self.cb_selector_color.currentIndex()]
        kwargs = {
            'color': color,
            'sl_type': self.cb_selector_type.currentText(),
            'width': self.plot_width.value()
        }
        self.sm.add_roi(**kwargs)
        return

    def apply_roi(self):
        self.sm.apply_roi()
        return 

    def show_location(self, event):
        # val = self.sm.show_location(event)
        # if val is not None:
        #     # self.lb_coordinate.setText(str(val))
        #     self.statusbar.showMessage(val)
        return


def run():
    # if os.name == 'nt':
    #     setup_windows_icon()
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    window = SimpleMaskGUI()
    app.exec_()


if __name__ == '__main__':
    run()


