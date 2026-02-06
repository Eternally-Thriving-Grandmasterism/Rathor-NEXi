# src/ml/gesture_transformer/train.py – Training loop stub v1.0
# GPU-ready, simple dataset, valence logging
# MIT License – Autonomicity Games Inc. 2026

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gesture_model import GestureTransformer

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Device: {device}")

    # Dummy data – replace with real dataset
    num_samples = 10000
    X = torch.randn(num_samples, args.seq_len, args.landmark_dim)
    y_gesture = torch.randint(0, args.num_classes, (num_samples,))
    y_valence = torch.rand(num_samples, args.future_horizon)
    dataset = TensorDataset(X, y_gesture, y_valence)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GestureTransformer(
        seq_len=args.seq_len,
        landmark_dim=args.landmark_dim,
        num_gesture_classes=args.num_classes,
        future_valence_horizon=args.future_horizon
    ).to(device)

    criterion_gesture = nn.CrossEntropyLoss()
    criterion_valence = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y_gesture, y_valence in loader:
            x, y_gesture, y_valence = x.to(device), y_gesture.to(device), y_valence.to(device)

            optimizer.zero_grad()
            gesture_logits, future_valence = model(x)

            loss_g = criterion_gesture(gesture_logits, y_gesture)
            loss_v = criterion_valence(future_valence, y_valence)
            loss = loss_g + loss_v

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        avg_valence = y_valence.mean().item()
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Avg valence: {avg_valence:.3f}")

    # Export
    dummy_input = torch.randn(1, args.seq_len, args.landmark_dim).to(device)
    model.export_to_onnx(dummy_input, args.output)

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
    main(args)
