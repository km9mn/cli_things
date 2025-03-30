#!/bin/sh

#SBATCH -J rtx_fetq_t5
#SBATCH -o rtx_feta_t5.out
#SBATCH -t 72:00:00

#### Select GPU
#SBATCH -p RTX6000ADA
#SBATCH --gres=gpu:4

## node 지정하기
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

cd  $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

echo "Start"
echo "conda PATH"

echo "source $HOME/anaconda3/etc/profile.d/conda.sh"
source $HOME/anaconda3/etc/profile.d/conda.sh

echo "conda activate fetaqa-e2e"
conda activate fetaqa-e2e

SAMPLES_DIR=$HOME/project/FeTaQA/end2end
cd $SAMPLES_DIR

python train.py configs/t5-large.json

date

echo "conda deactivate"

conda deactivate

squeue --job $SLURM_JOBID

echo "##### END #####"