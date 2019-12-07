# Anatomy Viewer

This is a simple viewer to display the 3D segmentation result of Bayesian CNNs.

<img src='figs/demo.gif' width='800px'>

## Requirements
- Python 3
- PyQt 5

## Getting Started
### Installation
- Install from this repository
```bash
git clone https://github.com/yuta-hi/anatomy-viewer
cd anatomy_viewer
python setup.py install
```

## Usage
```bash
muscle_viewer image.mhd label.mhd uncertainty.mhd
```

## Related repositories
- [bayesian_unet](https://github.com/yuta-hi/bayesian_unet)
