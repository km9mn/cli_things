#!/bin/sh

#SBATCH -J torhtune_0809
#SBATCH -o torchtune_0809.out
#SBATCH -t 72:00:00

#### Select GPU
#SBATCH -p A100-80GB
#SBATCH --gres=gpu:2

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

echo "conda activate torchtune"
conda activate torchtune

SAMPLES_DIR=$HOME/project/torchtune/
cd $SAMPLES_DIR
tune run --nproc_per_node 2 lora_finetune_distributed --config recipes/configs/llama3/cluster_70B_lora.yaml

cd $SLURM_SUBMIT_DIR

date

echo "conda deactivate torchtune"

conda deactivate torchtune

squeue --job $SLURM_JOBID

echo "##### END #####"