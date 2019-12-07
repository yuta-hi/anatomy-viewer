from __future__ import absolute_import

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal

import numpy as np
import cv2

from .utils import numpy_to_qpixmap

class ImageView(QtWidgets.QGraphicsView):

    windowSignal = pyqtSignal(float)
    levelSignal  = pyqtSignal(float)

    def __init__(self, *argv, **keywords):
        super().__init__(*argv, **keywords)

        # set scene
        image = QtWidgets.QGraphicsPixmapItem()
        scene = QtWidgets.QGraphicsScene(self)
        scene.addItem(image)
        self.setScene(scene)

        # set anchors
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

        # remove scroll bars
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # set background color
        _bg_color = (50, 50, 50)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(*_bg_color)))

        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setCacheMode(QtWidgets.QGraphicsView.CacheBackground)
        self.setRenderHints(QtGui.QPainter.Antialiasing |
            QtGui.QPainter.SmoothPixmapTransform |
            QtGui.QPainter.TextAntialiasing
        )

        self.image = image
        self.scene = scene
        self.zoom = None

        self.pressedMousePosition = None
        self.syncCenterViewList = []

    def hasImage(self):
        return not self.image.pixmap().isNull()

    def setImage(self, image):

        if isinstance(image, np.ndarray):
            qpixmap = numpy_to_qpixmap(image)

        if qpixmap and not qpixmap.isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self.image.setPixmap(qpixmap)
        else:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self.image.setPixmap(QtGui.QPixmap())

        if self.zoom is None:
            self.fitInView()

    def fitInView(self):

        if not self.hasImage():
            return

        rect = self.image.boundingRect()
        if rect.isNull():
            return

        self.setSceneRect(rect)

        unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
        self.scale(1. / unity.width(), 1. / unity.height())

        viewrect = self.viewport().rect()
        scenerect = self.transform().mapRect(rect)

        factor = min(viewrect.width() / scenerect.width(),
                        viewrect.height() / scenerect.height())

        self.scale(factor, factor)

        self.zoom = 0

    def setSyncCenter(self, view):
        if not isinstance(view, (list,tuple)):
            view = [view]
        for i in range(len(view)):
            assert isinstance(view[i], ImageView), '`view` should be `ImageView`..'
        self.syncCenterViewList = view

    def eventFilter(self, object, event):
        return QtWidgets.QWidget.eventFilter(self, object, event)

    def mousePressEvent(self, event):
        QtWidgets.QGraphicsView.mousePressEvent(self, event)
        self.pressedMousePosition = event.pos()

    def mouseReleaseEvent(self, event):
        QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)

    def mouseMoveEvent(self, event):
        QtWidgets.QGraphicsView.mouseMoveEvent(self, event)

        rightButton = event.buttons() == QtCore.Qt.RightButton
        leftButton  = event.buttons() == QtCore.Qt.LeftButton

        # window level
        if rightButton:
            dxPos = self.pressedMousePosition.x() - event.pos().x()
            dyPos = self.pressedMousePosition.y() - event.pos().y()

            if abs(dxPos) == abs(dyPos):
                pass
            elif abs(dxPos) > abs(dyPos):
                self.windowSignal.emit( np.sign(dxPos) )
            else:
                self.levelSignal.emit( np.sign(dyPos) )

        # center on
        if leftButton:
            for view in self.syncCenterViewList:
                view.centerOn(self.mapToScene(view.viewport().rect().center()))

    def wheelEvent(self, event):

        if not self.hasImage():
            return

        # update zoom ratio
        if event.angleDelta().y() > 0:
            factor = 1.25
            self.zoom += 1

        else:
            factor = 0.8
            self.zoom -= 1

        for view in self.syncCenterViewList:
            view.zoom = self.zoom

        # scaling
        if self.zoom > 0:
            self.scale(factor, factor)

            for view in self.syncCenterViewList:
                view.centerOn(self.mapToScene(view.viewport().rect().center()))
                view.scale(factor, factor)

        elif self.zoom == 0:
            self.fitInView()

            for view in self.syncCenterViewList:
                view.fitInView()
        else:
            # NOTE: do not allow the down scaling
            self.zoom = 0

