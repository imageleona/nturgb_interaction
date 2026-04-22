import csv
import os
import json
import argparse
import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import DEFAULT_NUM_FRAMES, INPUT_CHANNELS, KarateDataset, NTU_INTERACTION_CLASS_NAMES
from model import STGCN

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "json_output"))


def _save_history_plots(history: dict, save_dir: str):
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_graph.png"))
    plt.close()
    plt.figure()
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_graph.png"))
    plt.close()


def train_model():
    p = argparse.ArgumentParser(description="ST-GCN training (NTU RGB+D 120)")
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR, help="Root folder with *train.json files")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-interaction", action="store_true", help="Disable cross-person interaction edges")
    p.add_argument("--num-frames", type=int, default=None, metavar="T",
                   help=f"Target time steps after pad/resample (default: {DEFAULT_NUM_FRAMES})")
    args = p.parse_args()

    DATA_DIR = os.path.abspath(args.data_dir)
    print(f"Dataset directory: {DATA_DIR}")
    CLASSES = list(NTU_INTERACTION_CLASS_NAMES)
    USE_INTERACTION = not args.no_interaction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(_SCRIPT_DIR, "output", f"{timestamp}_training")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Device: {device} | Saving to: {save_dir}")

    full_dataset = KarateDataset(DATA_DIR, class_names=CLASSES, mode="train", num_frames=args.num_frames)
    if len(full_dataset) == 0:
        print("No training data found.")
        return

    in_ch = int(getattr(full_dataset, "in_channels", INPUT_CHANNELS))
    print(f"Model input channels: {in_ch}")

    n_val = int(len(full_dataset) * args.val_ratio)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = STGCN(
        num_classes=len(CLASSES), in_channels=in_ch, num_nodes=50,
        use_interaction=USE_INTERACTION,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 120], gamma=0.1)

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "in_channels": in_ch,
            "num_nodes": 50,
            "use_interaction": USE_INTERACTION,
            "classes": CLASSES,
            "num_frames": full_dataset.num_frames,
        }, f, indent=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    csv_file = open(os.path.join(save_dir, "epoch_log.csv"), "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(args.epochs):
        model.train()
        t_loss, t_corr, t_total = 0, 0, 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_corr += (out.argmax(1) == labels).sum().item()
            t_total += labels.size(0)

        model.eval()
        v_loss, v_corr, v_total = 0, 0, 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                out = model(data)
                v_loss += criterion(out, labels).item()
                v_corr += (out.argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        train_acc = 100 * t_corr / t_total
        val_acc = 100 * v_corr / v_total
        train_loss_avg = t_loss / len(train_loader)
        val_loss_avg = v_loss / len(val_loader)
        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        csv_writer.writerow([epoch + 1, f"{train_loss_avg:.6f}", f"{train_acc:.4f}", f"{val_loss_avg:.6f}", f"{val_acc:.4f}"])
        csv_file.flush()
        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        scheduler.step()

    csv_file.close()
    _save_history_plots(history, save_dir)
    print("Training Complete.")


if __name__ == "__main__":
    train_model()
