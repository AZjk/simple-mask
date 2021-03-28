from PyQt5 import QtCore
from simple_mask_ui import Ui_MainWindow as Ui
from simple_mask_kernel import SimpleMask
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
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

    def more_setup(self):
        self.btn_load.clicked.connect(self.load)
        self.btn_select.clicked.connect(self.select)
        self.btn_undo.clicked.connect(self.undo_select)
        self.btn_redo.clicked.connect(self.redo_select)
        ax = self.mp1.hdl.subplots(1, 2)
        self.canvas = self.mp1.hdl.fig.canvas
        self.ax = ax
        self.sm = SimpleMask(self.canvas, self.ax[0], self.ax[1])

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
        self.le_shape.setText(str(self.sm.shape))
        self.groupBox.repaint()

        self.sm.draw_roi()

        self.mp1.hdl.draw()
        self.mp1.parent().repaint()

    def select(self):
        if self.cid is None:
            print('create a selector')
            sl_type = self.cb_selector_type.currentText()
            self.mp1.hdl.setFocus()
            self.cid = self.mp1.hdl.mpl_connect("key_press_event", self.finish)
            self.sm.select(sl_type)
            self.btn_select.setText('Stop')
            self.cb_selector_type.setDisabled(True)
        else:
            event = KeyEvent('simulate enter', self.canvas, 'enter')
            self.finish(event)

    def finish(self, event):
        if self.cid is None:
            print('no active selector')
            return

        print('finish a selection')
        if event.key in ["escape", "enter"]:
            event.key = "escape"
            self.sm.finish(event)
            self.mp1.hdl.mpl_disconnect(self.cid)
            self.cid = None
            self.mp1.hdl.draw()
            self.mp1.parent().repaint()
            self.btn_select.setText('Start')
            self.cb_selector_type.setEnabled(True)

    def undo_select(self):
        if self.cid is not None:
            event = KeyEvent('simulate enter', self.canvas, 'enter')
            self.finish(event)
        self.sm.undo()
        self.mp1.hdl.draw()
        self.mp1.parent().repaint()

    def redo_select(self):
        if self.cid is not None:
            event = KeyEvent('simulate enter', self.canvas, 'enter')
            self.finish(event)
        self.sm.redo()
        self.mp1.hdl.draw()
        self.mp1.parent().repaint()

def run():
    # if os.name == 'nt':
    #     setup_windows_icon()
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    window = SimpleMaskGUI()
    app.exec_()


if __name__ == '__main__':
    run()


