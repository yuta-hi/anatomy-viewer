#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from anatomy_viewer import AnatomyViewerApp
from anatomy_viewer.utils import load_volume

_default_label_cmap = np.array([
    [0,0,0], [1,1,1], [1,1,1], [0,1,1], [0.75,1,0.25],
    [1,1,0], [0,1,0], [1,0.5,0.5], [1,0.5,0.5], [0.5,0,0.5],
    [0,0,1], [1,0,0], [1,0,1], [1,0.5,0], [0,1,1],
    [1,0,0], [1,1,0], [1,0.5,0], [1,0,1], [0,0,1],
    [0.5,0,0.5], [0,1,0], [0.5,0.5,0.5]])

_jet_mapper = plt.get_cmap('jet', 256)
_default_uncert_cmap = np.asarray([_jet_mapper(i)[:3] for i in range(256)])

def main():

    parser = argparse.ArgumentParser(description='Anatomy Viewer: Muscle',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('image',  type=str, help='Path to image file')
    parser.add_argument('label',  type=str, help='Path to label file')
    parser.add_argument('uncertainty', type=str, help='Path to uncertainty file')
    args = parser.parse_args()

    image, spacing = load_volume(args.image)
    label, _ = load_volume(args.label)
    uncert, _ = load_volume(args.uncertainty)

    app = QtWidgets.QApplication(sys.argv)
    main_window = AnatomyViewerApp(image,
                                   label, _default_label_cmap,
                                   uncert, _default_uncert_cmap,
                                   spacing)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
