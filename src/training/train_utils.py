#!/usr/bin/env python3
import os
import torch
from typing import Dict, Optional, Tuple

def get_peft_state_maybe_zero_3(named_params, bias):
    """
    Get the state dict of the PEFT model, handling Zero-3 optimization.
    """
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.replace("lora_A", "bias").replace("lora_B", "bias")
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """
    Get the state dict of the non-PEFT part of the model, handling Zero-3 optimization.
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param):
    """
    Handle DeepSpeed ZeRO-3 params.
    """
    if hasattr(param, "ds_id"):
        assert param.ds_status == 1  # Make sure the parameter is gathered
        param = param.data
    return param

def safe_save_model_for_hf_trainer(trainer, output_dir: str):
    """
    Save model safely for HF Trainer, handling DeepSpeed ZeRO-3.
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
