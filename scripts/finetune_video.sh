#!/bin/bash

MODEL_NAME="google/gemma-3-4b-it"

export PYTHONPATH=src:$PYTHONPATH

# It is strongly recommended to train Gemma3 models with the `eager` attention implementation instead of `flash_attention_2`

deepspeed src/training/train.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/video/data.json \
    --image_folder /path/to/your/video/folder \
    --max_num_frames 10 \
    --disable_flash_attn2 True \
    --lora_enable True \
    --vision_lora True \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --tune_img_projector True \
    --freeze_vision_tower True \
    --freeze_llm True \
    --bf16 True \
    --output_dir output/test_video \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4
