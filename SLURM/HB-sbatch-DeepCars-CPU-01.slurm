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

cd /hb/home/mamoghad/Scripts/deepcars_DQN_initial_version/

python DeepCars_DQN_Train.py
