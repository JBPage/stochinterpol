#!/bin/bash
#SBATCH --job-name=JBP_stochinterpol_train
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH -C a100
#SBATCH -A lgu@a100
#SBATCH --output=jeanzay_log/%j_%_output.txt
#SBATCH --error=jeanzay_log/%j_%_error.txt    

# Variables d'environnement CRITIQUES pour PyTorch DDP
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=2
export NCCL_DEBUG=INFO

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1 
export NCCL_SOCKET_IFNAME=^lo,docker0 
export NCCL_DEBUG=INFO 
export PYTHONFAULTHANDLER=1

cd $WORK/stochinterpol/
module load arch/a100
module load singularity/3.8.5
 

SIF_FILE="/lustre/fswork/projects/rech/lgu/ujg69ti/dockers/pytorch_lightning.sif"
DATA_DIR="/lustre/fswork/projects/rech/lgu/ujg69ti/RangeShifter_simulation/BASE/"
CACHE_DIR="/lustre/fswork/projects/rech/lgu/ujg69ti/cache/"
WORKDIR="/lustre/fswork/projects/rech/lgu/ujg69ti/stochinterpol"
SCRIPT_FILE="stochinterpol_training.py"
WANDB_API_KEY=41d83b795a926cd02f4b2215639352be85371226
CKPT_DIR="/lustre/fswork/projects/rech/lgu/ujg69ti/stochinterpol/checkpoints/stoch_interpolent_ConvNextBlock_pred_1"
#WEIGHTS_PATH="/lustre/fswork/projects/rech/lgu/ujg69ti/stochinterpol/checkpoints/DDPM_stoch_interpolent_ConvNextBlock_pred_1/last.ckpt"
command_to_run="export MASTER_ADDR=$MASTER_ADDR && \
	export MASTER_PORT=$MASTER_PORT && \
	export WORLD_SIZE=$WORLD_SIZE && \
	export NCCL_DEBUG=INFO && \
	export NCCL_P2P_DISABLE=1 && \
	export NCCL_IB_DISABLE=1 && \
	export NCCL_SOCKET_IFNAME=^lo,docker0 && \
	export WANDB_API_KEY=$WANDB_API_KEY && \
        export WANDB_DIR=$WORKDIR && \
        export WANDB_MODE=offline && \
        export DATA_DIR=$DATA_DIR && \
        export CUDA_LAUNCH_BLOCKING=1 && \
        export HDF5_USE_FILE_LOCKING=FALSE && \
        export CACHE_DIR=$CACHE_DIR && \
        export CKPT_DIR=$CKPT_DIR && \
        export WEIGHTS_PATH=$WEIGHTS_PATH && \
        python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())' && \
	python $WORKDIR/$SCRIPT_FILE \
            --project_name=stoch_interpolent \
            --run_name=almost_full_derivation \
            --run_mode='train' \
            --concat_mode='concat' \
            --epochs=50 \
            --cpus=1 \
            --lr_min=1e-5 \
            --lr_max=1e-4 \
            --latent_diffusion \
            --prediction_step=1 \
	    --ckpt_name='last.ckpt' \
            --lr_scheduler='plateau' \
            --train_ratio=0.6 \
            --validation_ratio=0.2 \
            --mixed_precision \
            --batch_size=25 \
            --nb_of_simulation_folders_train=250 \
            --nb_of_simulation_folders_valid=50 \
            --gradient_accumulation_steps=2 \
            "
start_container_cmd="srun singularity exec \
    --pwd /lustre/fswork/projects/rech/lgu/ujg69ti/stochinterpol \
    --bind /lustre/fswork/projects/rech/lgu/ujg69ti/:/lustre/fswork/projects/rech/lgu/ujg69ti/ \
    --bind /lustre/fswork/projects/rech/lgu/ujg69ti/stochinterpol:/lustre/fswork/projects/rech/lgu/ujg69ti/stochinterpol \
    --nv \
    /lustre/fsn1/singularity/images/ujg69ti/pytorch_lightning.sif"

echo "=================================="
echo "DEBUG 1"
echo "Working directory: $(pwd)"
echo "Container command will be: $command_to_run"
echo "================================================"

$start_container_cmd /bin/bash -c "$command_to_run"

echo "================================================"
echo "DEBUG: Singularity command finished with exit code: $?"
echo "================================================"
