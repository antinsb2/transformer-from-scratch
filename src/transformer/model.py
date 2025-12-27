"""
Complete transformer model.
"""

import torch.nn as nn
from .encoding import TokenEmbedding, PositionalEncoding
from .layers import TransformerEncoder


class SimpleTransformer(nn.Module):
    """
    Complete transformer model for next-token prediction.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        self.output = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Token indices [batch, seq_len]
            mask: Optional attention mask
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            attention_weights: List of attention weights from each layer
        """
        # Input processing
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer layers
        x, attention_weights = self.encoder(x, mask)
        
        # Output projection
        logits = self.output(x)
        
        return logits, attention_weights
