#!/bin/sh

#SBATCH -J wikisql_holi
#SBATCH -o wikisql_holi.out
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

echo "conda activate tune"
conda activate tune

SAMPLES_DIR=$HOME/project/SAIT_project/src/external/torchtune
cd $SAMPLES_DIR
tune run --nproc_per_node 4 --master_port 29600 full_finetune_distributed --config recipes/configs/qwen2_5/pieta/wikisql_holi.yaml

cd $SLURM_SUBMIT_DIR

date

echo "conda deactivate tune"

conda deactivate tune

squeue --job $SLURM_JOBID

echo "##### END #####"