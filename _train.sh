#!/bin/bash

# Load python virtualenv
virtualenv venv
source venv/bin/activate
pip3.6 install -r requirements.txt

# Set GPU device ID
export CUDA_VISIBLE_DEVICES=-1

# For MuJoCo
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Begin experiment
for SEED in {1..1}
do
    python3.6 main.py \
    --seed $SEED \
    --config "metaworld.yaml" \
    --agent-type "sac" \
    --prefix ""
done
