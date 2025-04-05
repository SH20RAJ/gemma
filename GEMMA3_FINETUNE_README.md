# Gemma3 Fine-tuning

This repository contains scripts for fine-tuning [Gemma3](https://huggingface.co/google/gemma-3-4b-it) models using Hugging Face.

## Features

- Full fine-tuning
- LoRA fine-tuning
- QLoRA support
- DeepSpeed integration
- Multi-image and video training
- Text-only data training
- Mixed-modality training

## Installation

### Requirements

- Ubuntu 22.04 (or compatible Linux distribution)
- NVIDIA GPU with CUDA support
- Python 3.8+

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.

### Example for text-only data

```json
[
  {
    "id": "000000033471",
    "conversations": [
      {
        "from": "human",
        "value": "Identify the odd one out: Twitter, Instagram, Telegram"
      },
      {
        "from": "gpt",
        "value": "Telegram"
      }
    ]
  }
]
```

### Example for single image dataset

```json
[
  {
    "id": "000000033471",
    "image": "000000033471.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      }
    ]
  }
]
```

### Example for multi-image dataset

```json
[
  {
    "id": "000000033471",
    "image": ["000000033471.jpg", "000000033472.jpg"],
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n<image>\nIs the perspective of the camera different?"
      },
      {
        "from": "gpt",
        "value": "Yes, the perspective of the camera is different."
      }
    ]
  }
]
```

### Example for video dataset

```json
[
  {
    "id": "sample1",
    "video": "sample1.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nWhat is going on in this video?"
      },
      {
        "from": "gpt",
        "value": "A man is walking down the road."
      }
    ]
  }
]
```

**Note:** Gemma3 processes videos as a sequence of images.

## Training

### Full Fine-tuning

```bash
bash scripts/finetune.sh
```

### Fine-tune with LoRA

If you want to train only the language model with LoRA and perform full training for the vision model:

```bash
bash scripts/finetune_lora.sh
```

If you want to train both the language model and the vision model with LoRA:

```bash
bash scripts/finetune_lora_vision.sh
```

### Train with video dataset

```bash
bash scripts/finetune_video.sh
```

### Merge LoRA Weights

```bash
bash scripts/merge_lora.sh
```

## Training Arguments

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Gemma3 model. **(Required)**
- `--optim` (str): Optimizer when training (default: `adamw_torch`).
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--tune_img_projector` (bool): Option to tune projector (default: True).
- `--num_lora_modules` (int): Number of target modules to add LoRA (-1 means all layers).
- `--vision_lr` (float): Learning rate for vision_model.
- `--projector_lr` (float): Learning rate for projector.
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--lora_enable` (bool): Option for enabling LoRA (default: False)
- `--vision_lora` (bool): Option for including vision_tower to the LoRA module. The `lora_enable` should be `True` to use this option. (default: False)
- `--lora_namespan_exclude` (str): Exclude modules with namespans to add LoRA.
- `--max_seq_length` (int): Maximum sequence length (default: 128K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--lora_rank` (int): LoRA rank (default: 64).
- `--lora_alpha` (int): LoRA alpha (default: 16).
- `--lora_dropout` (float): LoRA dropout (default: 0.05).
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).

**Note:** The learning rate of `vision_model` should be 10x ~ 5x smaller than the `language_model`.

## Known Issues

- If you encounter the error `Could not load library libcudnn_cnn_train.so.8`, run `unset LD_LIBRARY_PATH` to fix it.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
