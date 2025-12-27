"""
Feed-forward network for transformer.
"""

import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Args:
        d_model: Input/output dimension
        d_ff: Hidden layer dimension (typically 4 * d_model)
        dropout: Dropout probability
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            [batch, seq_len, d_model]
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
