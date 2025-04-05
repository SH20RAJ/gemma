#!/usr/bin/env python3
import os
import torch
import argparse
from peft import PeftModel
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

def merge_lora_weights(base_model_path, lora_model_path, output_path):
    """
    Merge LoRA weights into the base model.
    
    Args:
        base_model_path: Path to the base model
        lora_model_path: Path to the LoRA model
        output_path: Path to save the merged model
    """
    print(f"Loading base model from {base_model_path}")
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading LoRA model from {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    print("Merging weights")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}")
    model.save_pretrained(output_path)
    
    # Save processor
    processor = AutoProcessor.from_pretrained(base_model_path)
    processor.save_pretrained(output_path)
    
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into the base model")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--lora_model_path", type=str, required=True, help="Path to the LoRA model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model")
    
    args = parser.parse_args()
    
    merge_lora_weights(args.base_model_path, args.lora_model_path, args.output_path)

if __name__ == "__main__":
    main()
