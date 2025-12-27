"""
Attention mechanisms for transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        
    The input is split into num_heads, each processing d_k = d_model/num_heads dimensions.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Projections for all heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split into multiple heads: [batch, seq_len, d_model] -> [batch, heads, seq_len, d_k]"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine heads: [batch, heads, seq_len, d_k] -> [batch, seq_len, d_model]"""
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, 1, seq_len, seq_len]
            
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        # Project to Q, K, V
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32)
        )
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        # Combine heads and project
        output = self.combine_heads(output)
        output = self.W_o(output)
        
        return output, attention_weights
