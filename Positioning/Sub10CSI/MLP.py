# ===========================
# XY POSITIONING MLP TRAINING
# ===========================

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Dataset
# ---------------------------
class PositioningDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.float()
        self.y = y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------
# MLP Model
# ---------------------------
class MLPPositioning(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout)
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 2))  # (x, y)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Training Function
# ---------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    epochs=200,
    lr=1e-3,
    patience=20,
    device="cuda",
    save_path="best_model.pt"
):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )


    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    early_stop_counter = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                val_loss += criterion(preds, y).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f}"
        )

        # ---- Save Best Model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print("Best model saved")
        else:
            early_stop_counter += 1

        # ---- Early Stopping ----
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

    return train_losses, val_losses


# ---------------------------
# Plot Learning Curves
# ---------------------------
def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Test Evaluation
# ---------------------------
def evaluate_test_set(model, test_loader, device="cuda"):
    model.to(device)
    model.eval()

    criterion = nn.MSELoss()
    total_loss = 0.0

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)

            total_loss += criterion(preds, y).item()
            preds_all.append(preds.cpu())
            targets_all.append(y.cpu())

    mse = total_loss / len(test_loader)

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)

    rmse = torch.sqrt(((preds_all - targets_all) ** 2).mean()).item()

    print(f"\n Test MSE:  {mse:.6f}")
    print(f" Test RMSE: {rmse:.4f} meters")

    return mse, rmse


# ---------------------------
# Example Main Script
# ---------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Dummy Example Data ----
    N = 2000
    input_dim = 256  # flattened CSI / pilot vector

    X = torch.randn(N, input_dim)
    y = torch.rand(N, 2) * 10  # (x, y) in meters

    # ---- Train / Val / Test Split ----
    train_X, val_X, test_X = torch.split(X, [1400, 300, 300])
    train_y, val_y, test_y = torch.split(y, [1400, 300, 300])

    train_ds = PositioningDataset(train_X, train_y)
    val_ds   = PositioningDataset(val_X, val_y)
    test_ds  = PositioningDataset(test_X, test_y)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64)
    test_loader  = DataLoader(test_ds, batch_size=64)

    # ---- Model ----
    model = MLPPositioning(input_dim)

    # ---- Training ----
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        epochs=20,
        patience=15,
        device=device,
        save_path="best_mlp_positioning_csi_sub10"
    )

    # ---- Plot Learning Curves ----
    plot_learning_curves(train_losses, val_losses)

    # ---- Load Best Model ----
    model.load_state_dict(torch.load("best_mlp_positioning_csi_sub10", map_location=device))

    # ---- Test Evaluation ----
    evaluate_test_set(model, test_loader, device=device)
