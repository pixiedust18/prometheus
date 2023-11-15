#!/bin/bash
#SBATCH --job-name=llama2fsdp
#SBATCH --output=llama2fsdp.out
#SBATCH --error=llama2fsdp.err
#SBATCH --partition=babel-shared-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:A6000:2


# Your job commands go here
torchrun --nnodes 1 \
    --nproc_per_node 2 \
    llama_finetuning.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --batch_size_training 4  \
    --gradient_accumulation_steps 7 \
    --dist_checkpoint_root_folder "ckpt" \
    --dist_checkpoint_folder "fsdp" \
    --dataset feedback_collection_dataset \
    --data_file sample_train_data.json \
    --num_epochs 2 \
    --scheduler "step"  \
    --use_fast_kernel \
    --enable_fsdp \
    --pure_bf16 \
    --low_cpu_fsdp