#!/usr/bin/env python3
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

def load_model_and_processor(model_path, device="cuda"):
    """
    Load a fine-tuned Gemma3 model and processor.
    
    Args:
        model_path: Path to the fine-tuned model
        device: Device to load the model on
        
    Returns:
        model: The loaded model
        processor: The processor for the model
    """
    processor = AutoProcessor.from_pretrained(model_path)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    
    return model, processor
