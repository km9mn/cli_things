#!/bin/sh

#SBATCH -J qwen_wtq_k4_1
#SBATCH -o qwen_wtq_k4_1.out
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

MODEL_PATH="/home/kyuminkim/weights/llm/qwen_finetuned/qwen_wtq_k4/"
FOLD_PATH="/home/kyuminkim/datasets/table/pieta/results/qwen_wtq_kernel4_coord_1-3.json"
CSV_PATH="/home/kyuminkim/datasets/table/wtq/wikitq_test_csv"
TSV_PATH="/home/kyuminkim/datasets/table/wtq/wtq_test_split/wikitq_test_1-3.tsv"

HYDRA_FULL_ERROR=1 python src/test/inference/main_llama3.py evaluation_module=divide_conquer_vllm_qwen_wtq_coord_tensor_parallel evaluation_module.LLM_module.model_path="$MODEL_PATH" evaluation_module.LLM_module.temperature=0.7 evaluation_module.LLM_module.top_p=1 evaluation_module.LLM_module.tensor_parallel_size=4 evaluation_module.save_dir="$FOLD_PATH" evaluation_module.batch_size=null evaluation_module.dataset.csv_dir_path="$CSV_PATH" evaluation_module.dataset.tsv_path="$TSV_PATH" evaluation_module.dataset.kernel_size=4 evaluation_module.dataset.moving_size=1

date

echo "conda deactivate"

conda deactivate

squeue --job $SLURM_JOBID

echo "##### END #####"