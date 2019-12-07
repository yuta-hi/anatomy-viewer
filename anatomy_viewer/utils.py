from __future__ import absolute_import

from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np
import cv2
import SimpleITK as sitk

def load_volume(filename):

    itkimage = sitk.ReadImage(filename)
    volume = sitk.GetArrayFromImage(itkimage)
    spacing = np.array(itkimage.GetSpacing())

    if volume.ndim == 3:
        return volume.transpose(2,1,0), spacing
    else:
        raise ValueError('`volume.ndim` should be 3..')


def numpy_to_qpixmap(image):
    assert isinstance(image, np.ndarray), '`image` should be `np.ndarray`..'

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.ndim == 3:
        ch = image.shape[-1]
        if ch == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif ch == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    assert image.ndim == 3, '`image.ndim` should be 3..'
    assert image.shape[-1] == 4, '`image.shape[-1]` should be 4..'

    qimage = QtGui.QImage(image.data,
                          image.shape[1], image.shape[0],
                          image.shape[1] * 4,
                          QtGui.QImage.Format_ARGB32_Premultiplied)
    qpixmap = QtGui.QPixmap.fromImage(qimage)
    return qpixmap


def lut(label, cmap):
    assert np.max(label) <= len(cmap)
    cmap = 255.*cmap.copy()
    cmap = cmap.astype(np.uint8)
    cmap256 = np.zeros((256, 3), np.uint8)
    cmap256[:len(cmap)] = cmap
    im_r = cv2.LUT(label, cmap256[:, 2])  # NOTE: opencv's BGR format
    im_g = cv2.LUT(label, cmap256[:, 1])
    im_b = cv2.LUT(label, cmap256[:, 0])
    im_color = cv2.merge((im_r, im_g, im_b))
    return im_color


def clim(x, param, scale=255.):
    assert isinstance(param, (list, tuple))
    norm = (x.astype(np.float32) - param[0]) / (param[1] - param[0])
    return np.clip(norm, 0.0, 1.0, out=norm) * scale


def resize2d(x, spacing):
    return cv2.resize(x, None,
                      fx=1./spacing[0], fy=1./spacing[1],
                      interpolation=cv2.INTER_NEAREST)
