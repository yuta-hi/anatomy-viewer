from __future__ import absolute_import

from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np
import cv2
import six

from .image_view import ImageView
from .anatomy_viewer_ui import Ui_AnatomyViewer
from .utils import lut, clim, resize2d

imageStatistics = {
    'mean': lambda x: np.mean(x),
    'std': lambda x: np.std(x),
    'min': lambda x: np.min(x),
    'max': lambda x: np.max(x),
}

labeStatistics = {
    'min': lambda x: np.min(x),
    'max': lambda x: np.max(x),
}

uncertStatistics = {
    'mean': lambda x: np.mean(x),
    'std': lambda x: np.std(x),
    'min': lambda x: np.min(x),
    'max': lambda x: np.max(x),
}

mapSliceAxis = {
    'Axial': 2,
    'Coronal': 1,
    'Sagittal': 0,
}


def checkVolume(x, name):
    assert isinstance(x, np.ndarray), '%s should be `np.ndarray`..' % name
    assert x.ndim == 3, '%s.ndim should be 3..' % name


class AnatomyViewerApp(QtWidgets.QMainWindow):
    def __init__(self,
                 image,
                 label, label_cmap,
                 uncert, uncert_cmap,
                 spacing):

        super().__init__()

        checkVolume(image, 'image')
        checkVolume(label, 'label')
        checkVolume(uncert, 'uncert')

        assert image.shape == label.shape, 'image.shape != label.shape'
        assert image.shape == uncert.shape, 'image.shape != uncert.shape'

        self.imageVolume  = image
        self.labelVolume  = label
        self.uncertVolume = uncert

        self.labelColorMap  = label_cmap
        self.uncertColorMap = uncert_cmap

        self.volumeSpacing = spacing

        self.imageMean = np.mean(image)
        self.imageStd = np.std(image)
        self.uncertMean = np.mean(uncert)
        self.uncertStd = np.std(uncert)

        self.sliceIndex = 0
        self.sliceAxis = 'Axial'

        self.imageWindowLevel  = [1., 0.]
        self.uncertWindowLevel = [1., 0.]

        self.imageAlpha  = 0.2
        self.uncertAlpha = 0.2

        self.ui = None
        self.setupUi()
        self.setupTextBrowser()

    def setupUi(self):
        self.ui = Ui_AnatomyViewer()
        self.ui.setupUi(self)

        _margins = (0,0,0,0)

        # image
        self.viewerImage = ImageView(self.ui.graphicsViewImage)
        self.ui.viewerImage_layout = QtWidgets.QHBoxLayout()
        self.ui.viewerImage_layout.addWidget(self.viewerImage)
        self.ui.viewerImage_layout.setContentsMargins(*_margins)
        self.ui.graphicsViewImage.setLayout(self.ui.viewerImage_layout)

        # label
        self.viewerLabel = ImageView(self.ui.graphicsViewLabel)
        self.ui.viewerLabel_layout = QtWidgets.QHBoxLayout()
        self.ui.viewerLabel_layout.addWidget(self.viewerLabel)
        self.ui.viewerLabel_layout.setContentsMargins(*_margins)
        self.ui.graphicsViewLabel.setLayout(self.ui.viewerLabel_layout)

        # label overlay
        self.viewerLabelOverlay = ImageView(self.ui.graphicsViewLabelOverlay)
        self.ui.viewerLabelOverlay_layout = QtWidgets.QHBoxLayout()
        self.ui.viewerLabelOverlay_layout.addWidget(self.viewerLabelOverlay)
        self.ui.viewerLabelOverlay_layout.setContentsMargins(*_margins)
        self.ui.graphicsViewLabelOverlay.setLayout(self.ui.viewerLabelOverlay_layout)

        # uncertainty
        self.viewerUncert = ImageView(self.ui.graphicsViewUncert)
        self.ui.viewerUncert_layout = QtWidgets.QHBoxLayout()
        self.ui.viewerUncert_layout.addWidget(self.viewerUncert)
        self.ui.viewerUncert_layout.setContentsMargins(*_margins)
        self.ui.graphicsViewUncert.setLayout(self.ui.viewerUncert_layout)

        # uncertainty overlay
        self.viewerUncertOverlay = ImageView(self.ui.graphicsViewUncertOverlay)
        self.ui.viewerUncertOverlay_layout = QtWidgets.QHBoxLayout()
        self.ui.viewerUncertOverlay_layout.addWidget(self.viewerUncertOverlay)
        self.ui.viewerUncertOverlay_layout.setContentsMargins(*_margins)
        self.ui.graphicsViewUncertOverlay.setLayout(self.ui.viewerUncertOverlay_layout)

        # synchronization
        self.viewerImage.setSyncCenter([self.viewerLabel, self.viewerLabelOverlay, self.viewerUncert, self.viewerUncertOverlay])
        self.viewerLabel.setSyncCenter([self.viewerImage, self.viewerLabelOverlay, self.viewerUncert, self.viewerUncertOverlay])
        self.viewerLabelOverlay.setSyncCenter([self.viewerImage, self.viewerLabel, self.viewerUncert, self.viewerUncertOverlay])
        self.viewerUncert.setSyncCenter([self.viewerImage, self.viewerLabel, self.viewerLabelOverlay, self.viewerUncertOverlay])
        self.viewerUncertOverlay.setSyncCenter([self.viewerImage, self.viewerLabel, self.viewerLabelOverlay, self.viewerUncert])

        # connection
        self.ui.spinBoxSliceIndex.valueChanged[int].connect(self.setSliceIndex)
        self.ui.sliderSliceIndex.valueChanged[int].connect(self.slideSliceIndex)

        self.ui.doubleSpinBoxImageWindow.valueChanged[float].connect(self.setImageWindow)
        self.ui.doubleSpinBoxImageLevel.valueChanged[float].connect(self.setImageLevel)
        self.ui.doubleSpinBoxUncertWindow.valueChanged[float].connect(self.setUncertWindow)
        self.ui.doubleSpinBoxUncertLevel.valueChanged[float].connect(self.setUncertLevel)

        self.ui.doubleSpinBoxImageAlpha.valueChanged[float].connect(self.setImageAlpha)
        self.ui.doubleSpinBoxUncertAlpha.valueChanged[float].connect(self.setUncertAlpha)

        self.viewerImage.windowSignal[float].connect(self.addImageWindow)
        self.viewerImage.levelSignal[float].connect(self.addImageLevel)
        self.viewerUncert.windowSignal[float].connect(self.addUncertWindow)
        self.viewerUncert.levelSignal[float].connect(self.addUncertLevel)

        self.viewerImage.sliceSignal[float].connect(self.addSliceIndex)
        self.viewerLabel.sliceSignal[float].connect(self.addSliceIndex)
        self.viewerLabelOverlay.sliceSignal[float].connect(self.addSliceIndex)
        self.viewerUncert.sliceSignal[float].connect(self.addSliceIndex)
        self.viewerUncertOverlay.sliceSignal[float].connect(self.addSliceIndex)

        self.ui.comboBoxSliceAxis.activated[str].connect(self.setSliceAxis)

    def setupTextBrowser(self):

        # volume shape
        spacing = self.volumeSpacing
        shape_pix = self.imageVolume.shape
        shape_mm = np.asarray(shape_pix) * np.asarray(spacing)

        self.ui.textBrowserShape.append('size [pixel]: %d, %d, %d' % \
                                (shape_pix[0], shape_pix[1], shape_pix[2]))
        self.ui.textBrowserShape.append('size [mm]: %f, %f, %f' % \
                                (shape_mm[0], shape_mm[1], shape_mm[2]))
        self.ui.textBrowserShape.append('spacing: %f, %f, %f' % \
                                (spacing[0], spacing[1], spacing[2]))

        # scalar statistics
        for name, x, statistics in zip(['image', 'label', 'uncertainty'], \
                                    [self.imageVolume, self.labelVolume, self.uncertVolume], \
                                    [imageStatistics, labeStatistics, uncertStatistics]):
            self.ui.textBrowserScalar.append(name + ':')
            for function_name, function in six.iteritems(statistics):
                self.ui.textBrowserScalar.append('  %s: %f' % (function_name, function(x)))
            self.ui.textBrowserScalar.append('------')


    def show(self):
        super().show()

        nSlices = self.imageVolume.shape[mapSliceAxis[self.sliceAxis]]

        self.ui.spinBoxSliceIndex.setValue(nSlices//2)
        self.ui.spinBoxSliceIndex.setMinimum(0)
        self.ui.spinBoxSliceIndex.setMaximum(nSlices-1)

        self.ui.sliderSliceIndex.setValue(nSlices//2)
        self.ui.sliderSliceIndex.setRange(0, nSlices-1)
        self.ui.sliderSliceIndex.setTracking(True)

        self.ui.doubleSpinBoxImageWindow.setValue(3.0 * self.imageStd)
        self.ui.doubleSpinBoxImageWindow.setSingleStep(0.05 * self.imageStd)
        self.ui.doubleSpinBoxImageLevel.setValue(self.imageMean)
        self.ui.doubleSpinBoxImageLevel.setSingleStep(0.05 * self.imageStd)
        self.ui.doubleSpinBoxImageAlpha.setValue(self.imageAlpha)

        uncertMin, uncertMax = 0., np.percentile(self.uncertVolume, 99)
        self.ui.doubleSpinBoxUncertWindow.setValue(uncertMax - uncertMin)
        self.ui.doubleSpinBoxUncertWindow.setSingleStep(0.05 * self.uncertStd)
        self.ui.doubleSpinBoxUncertLevel.setValue((uncertMax - uncertMin)/2.)
        self.ui.doubleSpinBoxUncertLevel.setSingleStep(0.05 * self.uncertStd)
        self.ui.doubleSpinBoxUncertAlpha.setValue(self.uncertAlpha)

        self.update()

    def setSliceAxis(self, value):
        self.sliceAxis = value
        nSlices = self.imageVolume.shape[mapSliceAxis[value]]

        self.ui.spinBoxSliceIndex.setValue(nSlices//2)
        self.ui.spinBoxSliceIndex.setMinimum(0)
        self.ui.spinBoxSliceIndex.setMaximum(nSlices-1)

        self.ui.sliderSliceIndex.setValue(nSlices//2)
        self.ui.sliderSliceIndex.setRange(0, nSlices-1)
        self.ui.sliderSliceIndex.setTracking(True)

        self.update()

        # re-fit
        self.viewerImage.fitInView()
        self.viewerLabel.fitInView()
        self.viewerLabelOverlay.fitInView()
        self.viewerUncert.fitInView()
        self.viewerUncertOverlay.fitInView()

    def setImageWindow(self, value):
        self.imageWindowLevel[0] = value
        self.update()

    def setImageLevel(self, value):
        self.imageWindowLevel[1] = value
        self.update()

    def setImageAlpha(self, value):
        self.imageAlpha = value
        self.update()

    def addImageWindow(self, value):
        self.imageWindowLevel[0] += value * 0.05 * self.imageStd
        self.ui.doubleSpinBoxImageWindow.setValue(self.imageWindowLevel[0])

    def addImageLevel(self, value):
        self.imageWindowLevel[1] += value * 0.05 * self.imageStd
        self.ui.doubleSpinBoxImageLevel.setValue(self.imageWindowLevel[1])

    def setUncertWindow(self, value):
        self.uncertWindowLevel[0] = value
        self.update()

    def setUncertLevel(self, value):
        self.uncertWindowLevel[1] = value
        self.update()

    def addUncertWindow(self, value):
        self.uncertWindowLevel[0] += value * 0.05 * self.uncertStd
        self.ui.doubleSpinBoxUncertWindow.setValue(self.uncertWindowLevel[0])

    def addUncertLevel(self, value):
        self.uncertWindowLevel[1] += value * 0.05 * self.uncertStd
        self.ui.doubleSpinBoxUncertLevel.setValue(self.uncertWindowLevel[1])

    def setUncertAlpha(self, value):
        self.uncertAlpha = value
        self.update()

    def setSliceIndex(self, value):
        self.sliceIndex = value
        self.ui.sliderSliceIndex.setValue(value)
        self.update()

    def addSliceIndex(self, value):
        nSlices = self.imageVolume.shape[mapSliceAxis[self.sliceAxis]]
        self.sliceIndex = np.clip(self.sliceIndex + int(value), 0, nSlices - 1)
        self.ui.sliderSliceIndex.setValue(self.sliceIndex)
        self.ui.spinBoxSliceIndex.setValue(self.sliceIndex)
        self.update()

    def slideSliceIndex(self, value):
        self.sliceIndex = value
        self.ui.spinBoxSliceIndex.setValue(value)
        self.update()

    def update(self):

        # slicing
        if self.sliceAxis == 'Axial':
            imageSlice = self.imageVolume[:,:,self.sliceIndex].T
            labelSlice = self.labelVolume[:,:,self.sliceIndex].T
            uncertSlice = self.uncertVolume[:,:,self.sliceIndex].T
            spacing = (self.volumeSpacing[1], self.volumeSpacing[0])
        elif self.sliceAxis == 'Coronal':
            imageSlice = self.imageVolume[:,self.sliceIndex,::-1].T
            labelSlice = self.labelVolume[:,self.sliceIndex,::-1].T
            uncertSlice = self.uncertVolume[:,self.sliceIndex,::-1].T
            spacing = (self.volumeSpacing[2], self.volumeSpacing[0])
        elif self.sliceAxis == 'Sagittal':
            imageSlice = self.imageVolume[self.sliceIndex,:,::-1].T
            labelSlice = self.labelVolume[self.sliceIndex,:,::-1].T
            uncertSlice = self.uncertVolume[self.sliceIndex,:,::-1].T
            spacing = (self.volumeSpacing[2], self.volumeSpacing[1])

        # resize
        if spacing[0] != spacing[1]:
            imageSlice  = resize2d(imageSlice, spacing)
            labelSlice  = resize2d(labelSlice, spacing)
            uncertSlice = resize2d(uncertSlice, spacing)

        # image
        imageWindow, imageLevel = self.imageWindowLevel
        imageSlice = clim(imageSlice, (imageLevel - imageWindow/2., imageLevel + imageWindow/2.)).astype(np.uint8)
        imageSlice = cv2.cvtColor(imageSlice, cv2.COLOR_GRAY2BGR)

        # label
        labelSlice = lut(labelSlice.astype(np.uint8), self.labelColorMap)
        labelOverlaySlice = cv2.addWeighted(imageSlice, 1.0 - self.imageAlpha, labelSlice, self.imageAlpha, 0)

        # uncertainty
        uncertWindow, uncertLevel = self.uncertWindowLevel
        uncertSlice = clim(uncertSlice, (uncertLevel - uncertWindow/2., uncertLevel + uncertWindow/2.)).astype(np.uint8)
        uncertSlice = lut(uncertSlice, self.uncertColorMap)

        uncertOverlaySlice = cv2.addWeighted(imageSlice, 1.0 - self.uncertAlpha, uncertSlice, self.uncertAlpha, 0)

        # send to view
        self.viewerImage.setImage(imageSlice)
        self.viewerLabel.setImage(labelSlice)
        self.viewerLabelOverlay.setImage(labelOverlaySlice)
        self.viewerUncert.setImage(uncertSlice)
        self.viewerUncertOverlay.setImage(uncertOverlaySlice)
