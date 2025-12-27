"""
Transformer block and encoder stack.
"""

import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Single transformer block with:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Residual connections and layer normalization
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        # Self-attention with residual
        attn_output, attn_weights = self.attention(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual
        ff_output = self.ff(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Stack of transformer blocks.
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, 
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, d_model]
            all_attention_weights: List of attention weights from each layer
        """
        attention_weights_all = []
        
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            attention_weights_all.append(attention_weights)
        
        return x, attention_weights_all
