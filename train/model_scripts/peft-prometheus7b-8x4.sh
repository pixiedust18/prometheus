#!/bin/bash
#SBATCH --job-name=prometheus-peft
#SBATCH --output=prometheus-peft.out
#SBATCH --error=prometheus-peft.err
#SBATCH --partition=babel-shared-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:A6000:1


# Your job commands go here
python -m llama_finetuning \
 --use_peft \
 --peft_method lora \
 --quantization \
 --model_name "kaist-ai/prometheus-7b-v1.0" \
 --batch_size_training 8 \
 --gradient_accumulation_steps 4\
 --dataset "feedback_collection_dataset" \
 --data_file sample_train_data.json \
 --num_epochs 1 \
 --scheduler "step" \
 --output_dir "ckpt/peft-prometheus-8x4" 