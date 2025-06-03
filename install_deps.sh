#!/bin/bash

sudo apt-get update
sudo apt-get install -y cmake libopenmpi-dev zlib1g-dev

# you should keep a newest pip version
pip install "gym==0.19.0"
conda install tensorflow==1.14
pip install stable-baselines ray
pip install stable-baselines[mpi]