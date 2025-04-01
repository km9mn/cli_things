#!/bin/sh

#SBATCH -J hitab_3-4
#SBATCH -o hitab_3-4.out
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

echo "conda activate sait"
conda activate sait

SAMPLES_DIR=$HOME/project/SAIT_project/
cd $SAMPLES_DIR

SPLIT="3-4"
MODEL_PATH="/home/kyuminkim/weights/llama/llama3_tuned/wtq_kernel4_coord_after_balanced_epoch_0"
FOLD_PATH="/home/kyuminkim/datasets/table/pieta/results/llama_hitab_kernel4_coord_$SPLIT.json"
CSV_PATH="/home/kyuminkim/datasets/table/hitab/hitab_test_csv"
JSON_PATH="/home/kyuminkim/datasets/table/hitab/hitab_test.json"

HYDRA_FULL_ERROR=1 python src/test/inference/main_llama3.py evaluation_module=divide_conquer_vllm_llama3_wikisql_coord_tensor_parallel_split evaluation_module.LLM_module.model_path="$MODEL_PATH" evaluation_module.LLM_module.temperature=0.7 evaluation_module.LLM_module.top_p=1 evaluation_module.LLM_module.tensor_parallel_size=4 evaluation_module.save_dir="$FOLD_PATH" evaluation_module.batch_size=NULL evaluation_module.dataset.json_path="$JSON_PATH" evaluation_module.dataset.kernel_size=4 evaluation_module.dataset.csv_dir="$CSV_PATH" evaluation_module.dataset.moving_size=1 evaluation_module.dataset.split="$SPLIT"

date

echo "conda deactivate"

conda deactivate

squeue --job $SLURM_JOBID

echo "##### END #####"