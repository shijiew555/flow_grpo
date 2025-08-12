import torch
import os
import json
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict


def calculate_zero_std_ratio(std_values):
    total_count = std_values.numel()
    zero_count = (std_values == 0).sum().item()
    zero_ratio = zero_count / total_count
    return zero_ratio


def unwrap_model(model):
    model = model._orig_mod if is_compiled_module(model) else model
    model = model.module if hasattr(model, "module") else model
    return model


def save_ckpt(ckpt_name, log_dir, model, accelerator, config, optimizer=None):
    unwrapped_model = unwrap_model(model)
    
    # Save LoRA checkpoint
    peft_model_state_dict = get_peft_model_state_dict(unwrapped_model)
    ckpt_path = os.path.join(log_dir, ckpt_name)
    
    torch.save({
        "model_state_dict": peft_model_state_dict,
        "config": config
    }, ckpt_path)
    
    if accelerator.is_main_process:
        print(f"Checkpoint saved: {ckpt_path}")


def create_generator(config, batch_size=1, num_images_per_prompt=None):
    if config.sample.seed is None:
        generator = None
    else:
        if num_images_per_prompt is None:
            num_images_per_prompt = config.sample.num_images_per_prompt
        # Create different generators for each image
        generator = torch.Generator(device="cpu").manual_seed(config.sample.seed)
    return generator


def get_dataset_class_for_prompt_fn(prompt_fn):
    """Get dataset class based on prompt function, matching original script logic."""
    from .datasets import TextPromptDataset, GenevalPromptDataset, GenevalPromptImageDataset
    
    if prompt_fn == "general_ocr":
        return TextPromptDataset
    elif prompt_fn == "geneval":
        return GenevalPromptDataset
    else:
        # Default to TextPromptDataset for other prompt functions
        return TextPromptDataset