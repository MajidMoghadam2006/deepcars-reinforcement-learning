#!/bin/bash
#   Note - there can be no empty lines between #SBATCH directives.
#SBATCH --job-name=deepcars-init_ver
#SBATCH --output=cpu_%j.out
#SBATCH --error=cpu_%j.err
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mamoghad@ucsc.edu
#SBATCH --partition=128x24

source /hb/home/mamoghad/Scripts/deepcars_DQN_initial_version/venv/bin/activate

# This sets the virtual env path on top of python modules path to load the tensorflow from venv instead of python module env.
export PYTHONPATH=/hb/home/mamoghad/Scripts/deepcars_DQN_initial_version/venv/lib/python3.6/site-packages:$PYTHONPATH

cd /hb/home/mamoghad/Scripts/deepcars_DQN_initial_version/

python Train_DeepCars_DDQN.py
