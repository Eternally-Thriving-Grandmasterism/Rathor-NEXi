# src/ml/gesture_transformer/train_full.py – Full Training Loop Integration v1.0
# Data loading + augmentation + model + optimizer + logging + checkpointing + mercy gating + ONNX export
# Bazel-ready, CUDA-aware, valence-modulated early stopping
# MIT License – Autonomicity Games Inc. 2026

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from gesture_model import GestureTransformer
from data_augmentation import GestureAugmentation
from torch.utils.tensorboard import SummaryWriter

def compute_valence_projection(model, loader, device, horizon=10):
    """Simple forward projection of future valence over horizon steps"""
    model.eval()
    with torch.no_grad():
        projections = []
        for x, _, _ in loader:
            x = x.to(device)
            _, future_v = model(x)
            projections.append(future_v.mean(dim=0).cpu().numpy())
            if len(projections) >= 100: break  # cap for speed
    return torch.tensor(projections).mean(dim=0).mean().item()  # avg projected valence

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Device: {device} | Valence: {currentValence.get():.3f}")

    # ─── Data ────────────────────────────────────────────────────────
    # Dummy dataset – replace with real landmark loader
    num_samples = 20000
    X = torch.randn(num_samples, args.seq_len, args.landmark_dim)
    y_gesture = torch.randint(0, args.num_classes, (num_samples,))
    y_valence = torch.rand(num_samples, args.future_horizon)
    dataset = TensorDataset(X, y_gesture, y_valence)

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ─── Augmentation ────────────────────────────────────────────────
    aug = GestureAugmentation(
        seq_len=args.seq_len,
        landmark_dim=args.landmark_dim,
        p_spatial_noise=0.5,
        p_temporal_dropout=0.3,
        p_time_warp=0.4,
        p_rotation=0.35,
        p_scaling=0.35,
        p_gaussian_noise=0.6
    ).to(device)

    # ─── Model ───────────────────────────────────────────────────────
    model = GestureTransformer(
        seq_len=args.seq_len,
        landmark_dim=args.landmark_dim,
        num_gesture_classes=args.num_classes,
        future_valence_horizon=args.future_horizon
    ).to(device)

    # ─── Loss & Optimizer ────────────────────────────────────────────
    criterion_gesture = nn.CrossEntropyLoss()
    criterion_valence = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard logging
    writer = SummaryWriter(log_dir=f"runs/gesture_transformer_{int(time.time())}")

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_g_loss = 0
        total_v_loss = 0

        for x, y_gesture, y_valence in train_loader:
            x, y_gesture, y_valence = x.to(device), y_gesture.to(device), y_valence.to(device)

            # Apply augmentation (valence-modulated)
            x = aug(x, valence=currentValence.get())

            optimizer.zero_grad()
            gesture_logits, future_valence = model(x)

            loss_g = criterion_gesture(gesture_logits, y_gesture)
            loss_v = criterion_valence(future_valence, y_valence)
            loss = loss_g + loss_v

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_g_loss += loss_g.item()
            total_v_loss += loss_v.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)
        avg_v_loss = total_v_loss / len(train_loader)

        # Validation projection check
        proj_valence = compute_valence_projection(model, val_loader, device)
        is_safe = proj_valence >= currentValence.get() - 0.05

        if not is_safe:
            print(f"[Mercy Gate] Epoch {epoch+1} – Projected valence drop too high ({proj_valence:.3f}) – skipping checkpoint")
            continue

        # Checkpoint if best
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"checkpoints/gesture_transformer_best.pt")
            print(f"[Checkpoint] New best at epoch {epoch+1} – loss {avg_loss:.4f}")

        # Logging
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | G: {avg_g_loss:.4f} | V: {avg_v_loss:.4f} | Proj V: {proj_valence:.3f}")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Loss/gesture", avg_g_loss, epoch)
        writer.add_scalar("Loss/valence", avg_v_loss, epoch)
        writer.add_scalar("Valence/projected", proj_valence, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)

    # Final export
    dummy_input = torch.randn(1, args.seq_len, args.landmark_dim).to(device)
    model.load_state_dict(torch.load("checkpoints/gesture_transformer_best.pt"))
    model.export_to_onnx(dummy_input, "gesture_transformer_final.onnx")

    writer.close()
    print(f"[Training] Complete – best loss {best_val_loss:.4f} at epoch {best_epoch+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq_len", type=int, default=45)
    parser.add_argument("--landmark_dim", type=int, default=225)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--future_horizon", type=int, default=10)
    args = parser.parse_args()
    main(args)
