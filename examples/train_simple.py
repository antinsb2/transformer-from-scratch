"""
Simple training example for transformer model.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from transformer import SimpleTransformer, count_parameters


def create_toy_dataset(num_samples=1000, seq_len=10, vocab_size=20):
    """Create simple sequence prediction dataset."""
    X = torch.randint(1, vocab_size, (num_samples, seq_len))
    y = torch.roll(X, -1, dims=1)
    return X, y


def train():
    """Train transformer on toy task."""
    # Hyperparameters
    vocab_size = 20
    d_model = 128
    num_layers = 2
    num_heads = 4
    d_ff = 512
    batch_size = 32
    num_epochs = 50
    lr = 0.001
    
    # Create model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create dataset
    X_train, y_train = create_toy_dataset(num_samples=1000)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nTraining started...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            logits, _ = model(batch_X)
            loss = criterion(logits.view(-1, vocab_size), batch_y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\n✅ Training complete!")
    
    # Test
    model.eval()
    with torch.no_grad():
        test_X = X_train[0:1]
        test_y = y_train[0:1]
        logits, _ = model(test_X)
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions[0] == test_y[0]).float().mean()
        
        print(f"\nTest accuracy: {accuracy:.1%}")
        print(f"Input:     {test_X[0].tolist()}")
        print(f"Predicted: {predictions[0].tolist()}")
        print(f"Target:    {test_y[0].tolist()}")


if __name__ == "__main__":
    train()
