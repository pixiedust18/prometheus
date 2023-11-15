#!/bin/bash
#SBATCH --job-name=fsdp2hf
#SBATCH --output=fsdp2hf.out
#SBATCH --error=fsdp2hf.err
#SBATCH --partition=babel-shared-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:A6000:1


# Your job commands go here
python -m convert_fsdp_to_hf \
  --fsdp_checkpoint_path ckpt/fsdp-4x7-kaist-ai/prometheus-7b-v1.0 \
  --consolidated_model_path ckpt/fsdp-prometheus7b-4x7 \
  --HF_model_path_or_name "kaist-ai/prometheus-7b-v1.0" \
  --repo_id "anlp-csk/fsdp-prometheus7b-4x7"