#!/usr/bin/env python

from PyQt5 import uic
import os

if __name__ == '__main__':

    if os.path.dirname(__file__) != '':
        os.chdir(os.path.dirname(__file__))

    with open('anatomy_viewer.ui') as ui_file:
        with open('anatomy_viewer_ui.py', 'w') as py_ui_file:
            uic.compileUi(ui_file, py_ui_file)
