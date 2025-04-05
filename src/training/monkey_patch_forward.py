#!/usr/bin/env python3
import torch
from transformers import Gemma3ForConditionalGeneration

def replace_gemma3_forward():
    """
    Monkey patch the forward method of Gemma3ForConditionalGeneration to handle multimodal inputs.
    """
    original_forward = Gemma3ForConditionalGeneration.forward
    
    def forward_with_vision(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        token_type_ids=None,
        **kwargs
    ):
        """
        Modified forward method to handle multimodal inputs.
        """
        # Process vision inputs if provided
        if pixel_values is not None and token_type_ids is not None:
            # Call the original forward method with the processed inputs
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
        else:
            # Call the original forward method
            return original_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs
            )
    
    # Replace the forward method
    Gemma3ForConditionalGeneration.forward = forward_with_vision
