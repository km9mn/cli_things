#!/bin/sh

#SBATCH -J tune-kernel
#SBATCH -o tune-kernel.out
#SBATCH -t 72:00:00

#### Select GPU
#SBATCH -p A100-80GB
#SBATCH --gres=gpu:2

## node 지정하기
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1

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

echo "conda activate tune"
conda activate tune

SAMPLES_DIR=$HOME/project/SAIT_project/src/external/torchtune
bash $SAMPLES_DIR/recipes/scripts/cluster_tune_lora_kernel_filtered_123_dulda.sh

date

echo "conda deactivate tune"

conda deactivate tune

squeue --job $SLURM_JOBID

echo "##### END #####"