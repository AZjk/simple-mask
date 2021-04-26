import pyqtgraph as pg
from mpl_cmaps_in_ImageItem import pg_get_cmap
import matplotlib.pyplot as plt


class ImageViewROI(pg.ImageView):
    def __init__(self, *arg, **kwargs):
        super(ImageViewROI, self).__init__(*arg, **kwargs)
        self.removeItem(self.roi)
        self.roi = []
        self.ui.roiBtn.setDisabled(True)
        self.ui.menuBtn.setDisabled(True)
    
    def roiClicked(self):
        pass

    def roiChanged(self):
        pass

    def adjust_viewbox(self):
        vb = self.getView()
        xMin, xMax = vb.viewRange()[0]
        yMin, yMax = vb.viewRange()[1]

        vb.setLimits(xMin=xMin,
                     xMax=xMax,
                     yMin=yMin,
                     yMax=yMax,
                     minXRange=(xMax - xMin) / 50,
                     minYRange=(yMax - yMin) / 50)
        vb.setMouseMode(vb.RectMode)
        vb.setAspectLocked(1.0)

    def reset_limits(self):
        """
        reset the viewbox's limits so updating image won't break the layout;
        """
        self.view.state['limits'] = {'xLimits': [None, None],
                                     'yLimits': [None, None],
                                     'xRange': [None, None],
                                     'yRange': [None, None]
                                     } 

    def set_colormap(self, cmap):
        pg_cmap = pg_get_cmap(plt.get_cmap(cmap))
        self.setColorMap(pg_cmap)
    
    def clear(self):
        for t in self.roi:
            self.remove_item(t)

        super(ImageViewROI, self).clear()
        self.reset_limits()
        # incase the signal isn't connected to anything.
        try:
            self.scene.sigMouseMoved.disconnect()
        except:
            pass

    def add_item(self, t):
        self.roi.append(t)
        self.addItem(t)

    def remove_item(self, t):
        self.roi.remove(t)
        self.removeItem(t)