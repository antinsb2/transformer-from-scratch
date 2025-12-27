"""
Utility functions for transformer model.
"""

import torch


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create causal mask for autoregressive generation.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Mask tensor [1, 1, seq_len, seq_len]
        True = can attend, False = cannot attend
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = (mask == 0).unsqueeze(0).unsqueeze(0)
    return mask


def count_parameters(model) -> int:
    """
    Count total trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_padding_mask(seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create padding mask for variable-length sequences.
    
    Args:
        seq_lengths: Actual lengths [batch_size]
        max_len: Maximum sequence length
        
    Returns:
        Mask [batch_size, 1, 1, max_len]
    """
    batch_size = seq_lengths.size(0)
    mask = torch.arange(max_len).expand(batch_size, max_len) < seq_lengths.unsqueeze(1)
    return mask.unsqueeze(1).unsqueeze(2)
