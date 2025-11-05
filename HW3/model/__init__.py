"""Model architectures for music generation"""

from .model_transformers import GPT2, TransformerXL, create_model

__all__ = ['GPT2', 'TransformerXL', 'create_model']
