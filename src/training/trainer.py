#!/usr/bin/env python3
import torch
from transformers import Trainer
from typing import Dict, List, Optional, Union, Any

class Gemma3Trainer(Trainer):
    def __init__(self, processor=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom compute_loss method to handle multimodal inputs
        """
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare inputs for the model, sending them to the correct device
        """
        inputs = super()._prepare_inputs(inputs)
        
        # Handle pixel_values separately
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.args.device)
            
        return inputs
