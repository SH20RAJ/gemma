#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --base_model_path "google/gemma-3-4b-it" \
    --lora_model_path "output/test_lora" \
    --output_path "output/merged_model"
