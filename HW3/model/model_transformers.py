"""
model.py - Music generation model architectures
"""

from transformers import (
    GPT2Config, 
    GPT2LMHeadModel,
    TransfoXLConfig,
    TransfoXLLMHeadModel
)
import torch.nn as nn
import torch


class GPT2(nn.Module):    

    def __init__(
        self,
        vocab_size,
        max_seq_len=1024,
        n_embd=512,
        n_layer=12,
        n_head=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    ):
        super().__init__()
        
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_len,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_embd * 4,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        
        self.model = GPT2LMHeadModel(self.config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class TransformerXL(nn.Module):
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layer=12,
        n_head=8,
        mem_len=512,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    ):
        super().__init__()
        
        self.config = TransfoXLConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            d_embed=d_model,
            n_layer=n_layer,
            n_head=n_head,
            d_head=d_model // n_head,
            d_inner=d_model * 4,
            dropout=0.1,
            dropatt=0.1,
            mem_len=mem_len,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        
        self.model = TransfoXLLMHeadModel(self.config)
    
    def forward(self, input_ids, attention_mask=None, labels=None, mems=None):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            mems=mems
        )
    
    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def create_model(model_type, vocab_size, **kwargs):

    """
    Factory function to create models
    
    Args:
        model_type: 'gpt2' or 'transformer-xl'
        vocab_size: Size of vocabulary
        **kwargs: Model-specific arguments
    
    Returns:
        Model instance
    """
    if model_type == 'gpt2':
        return GPT2(vocab_size=vocab_size, **kwargs)
    elif model_type == 'transformer-xl':
        return TransformerXL(vocab_size=vocab_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def estimate_gpu_memory(model, batch_size, seq_len, dtype=torch.float32):
    """
    Estimate GPU memory usage for a model.

    Args:
        model: The PyTorch model.
        batch_size: Batch size for input.
        seq_len: Sequence length for input.
        dtype: Data type (default: torch.float32).

    Returns:
        Estimated memory usage in MB.
    """
    # Calculate parameter memory
    param_memory = sum(p.numel() for p in model.parameters()) * torch.finfo(dtype).bits / 8 / 1e6

    # Calculate activation memory (forward pass)
    activation_memory = batch_size * seq_len * model.config.d_model * torch.finfo(dtype).bits / 8 / 1e6

    # Total memory (parameters + activations + gradients)
    total_memory = param_memory + 2 * activation_memory  # Gradients require same memory as activations

    return total_memory

if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test GPT-2
    model = create_model(
        model_type='gpt2',
        vocab_size=5000,
        n_layer=12,
        n_embd=512,
        n_head=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    print(f"GPT-2 Model created: {model.num_parameters():,} parameters")
    gpt2_memory = estimate_gpu_memory(model, batch_size=32, seq_len=1024)
    print(f"Estimated GPU memory for GPT-2: {gpt2_memory:.2f} MB")
    
    # Test Transformer-XL
    model_xl = create_model(
        model_type='transformer-xl',
        vocab_size=10000,
        n_layer=12,
        d_model=512,
        n_head=8,
        cutoffsq=512,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    print(f"Transformer-XL Model created: {model_xl.num_parameters():,} parameters")
    transformer_xl_memory = estimate_gpu_memory(model_xl, batch_size=32, seq_len=512)
    print(f"Estimated GPU memory for Transformer-XL: {transformer_xl_memory:.2f} MB")