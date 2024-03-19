#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install PyTorch with specific CUDA version
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
pip install tensorboard==2.11.2
pip install PyYAML==6.0
pip install selfies==0.2.4
pip install rdkit==2022.9.5
pip install PyTDC==0.4.0
pip install networkx==2.6.3
pip install pandas==1.3.5
