#!/bin/bash

sudo apt-get update
sudo apt-get install -y cmake libopenmpi-dev zlib1g-dev

# you should keep a newest pip version
sudo pip3 install gym tensorflow==1.14.0
sudo pip3 install stable-baselines ray
sudo pip3 install stable-baselines[mpi]