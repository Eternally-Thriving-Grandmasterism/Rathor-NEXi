# src/ml/gesture_transformer/train.py – Training loop stub v1.1
# GPU-ready, simple dataset, valence logging
# MIT License – Autonomicity Games Inc. 2026

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gesture_model import GestureTransformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Device: {device}")

    # Hyperparams
    batch_size = 64
    epochs = 100
    lr = 1e-4
    seq_len = 45
    landmark_dim = 225

    # Dummy dataset (replace with real data loader)
    num_samples = 10000
    X = torch.randn(num_samples, seq_len, landmark_dim)
    y_gesture = torch.randint(0, 5, (num_samples,))
    y_valence = torch.rand(num_samples, 10)  # future 10 steps
    dataset = TensorDataset(X, y_gesture, y_valence)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = GestureTransformer(
        seq_len=seq_len,
        landmark_dim=landmark_dim,
        num_gesture_classes=5,
        future_valence_horizon=10
    ).to(device)

    # Loss & optimizer
    criterion_gesture = nn.CrossEntropyLoss()
    criterion_valence = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in loader:
            x, y_gesture, y_valence = [t.to(device) for t in batch]

            optimizer.zero_grad()
            gesture_logits, future_valence = model(x)

            loss_g = criterion_gesture(gesture_logits, y_gesture)
            loss_v = criterion_valence(future_valence, y_valence)
            loss = loss_g + loss_v

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        current_valence = y_valence.mean().item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Avg valence: {current_valence:.3f}")

    # Export
    dummy_input = torch.randn(1, seq_len, landmark_dim).to(device)
    model.export_to_onnx(dummy_input, "gesture_transformer.onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=45)
    parser.add_argument("--landmark_dim", type=int, default=225)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--future_horizon", type=int, default=10)
    parser.add_argument("--output", type=str, default="gesture_transformer.onnx")
    args = parser.parse_args()
    main()
