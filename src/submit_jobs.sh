#!/bin/bash
#SBATCH -A MASCOLO-SL2-GPU
#SBATCH --output=/home/hy381/model_training/outputs/%x_%j.out  #! Saves output as job_name_jobID.out
#SBATCH --nodes=1              #! node count
#SBATCH --ntasks=4             #! total number of tasks across all node
#SBATCH --time=10:00:00
#SBATCH --mail-type=begin        #! send email when job begins
#SBATCH --mail-type=end          #! send email when job ends
#SBATCH --mail-type=fail         #! send email if job fails
#SBATCH --mail-user=hy381@cam.ac.uk
#SBATCH --gres=gpu:4
#SBATCH -p ampere

module purge
module load gcc/8
module load rhel8/default-gpu
module unload cuda/8.0
module load cuda/12.1  

conda init
source /home/hy381/model_training/tf_env/bin/activate
python "$1"