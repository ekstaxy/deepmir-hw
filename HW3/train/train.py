import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.dataset import Dataset_Pop1K7
from model.model_transformers import GPT2, TransformerXL

def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x  # Placeholder for custom collate function if needed
    )

def create_optimizer(model, lr=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr)

def create_model(model_type='gpt2', **kwargs):
    if model_type == 'gpt2':
        from model.model_transformers import GPT2
        model = GPT2(**kwargs)
    elif model_type == 'transformer-xl':
        from model.model_transformers import TransformerXL
        model = TransformerXL(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model