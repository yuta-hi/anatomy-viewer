#!/usr/bin/env python

from setuptools import find_packages
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='anatomy_viewer',
    version='1.0.0',
    description='Anatomy viewer for BCNNs',
    long_description=open('README.md').read(),
    author='yuta-hi',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'muscle_viewer=scripts.muscle_viewer:main',
        ]
    },
    install_requires=open('requirements.txt').readlines(),
    url='https://github.com/yuta-hi/anatomy-viewer',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
