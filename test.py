import os
import json
import argparse
import datetime
from typing import Optional

import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from dataset import KarateDataset
from model import STGCN


def _interaction_mode_from_cfg(cfg: dict) -> str:
    m = cfg.get("interaction_mode")
    if m in ("none", "full", "hand_cross"):
        return m
    return "full" if cfg.get("use_interaction", True) else "none"


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "json_output")
)
_OUTPUT_ROOT = os.path.join(_SCRIPT_DIR, "output")


def _find_latest_stgcn_training_dir() -> Optional[str]:
    """Newest output/*_training with config.json and best_model.pth (ST-GCN)."""
    if not os.path.isdir(_OUTPUT_ROOT):
        return None
    best_path, best_mtime = None, -1.0
    for name in os.listdir(_OUTPUT_ROOT):
        if not name.endswith("_training"):
            continue
        path = os.path.join(_OUTPUT_ROOT, name)
        if not os.path.isdir(path):
            continue
        if not (
            os.path.isfile(os.path.join(path, "config.json"))
            and os.path.isfile(os.path.join(path, "best_model.pth"))
        ):
            continue
        m = os.path.getmtime(path)
        if m > best_mtime:
            best_mtime, best_path = m, path
    return best_path


def _save_confusion_matrix(cm, class_names, save_path, txt_path):
    cm = np.asarray(cm, dtype=np.float64)
    n = len(class_names)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Test confusion matrix (row-normalized)")
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val > 0:
                text_color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    col_w = max(len(c) for c in class_names) + 2
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in class_names)
    lines = [header]
    for i, row_name in enumerate(class_names):
        row = f"{row_name:<{col_w}}" + "".join(f"{cm_norm[i, j]:>{col_w}.2f}" for j in range(n))
        lines.append(row)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def test_model():
    p = argparse.ArgumentParser(description="ST-GCN test (NTU RGB+D 120)")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Training run folder with config.json and best_model.pth; "
        "default: latest output/*_training from training.py",
    )
    p.add_argument(
        "--data-dir",
        default=_DEFAULT_DATA_DIR,
        help="Root folder with *test.json",
    )
    p.add_argument("--wandb-project", default="nturgb-interaction", help="W&B project name")
    p.add_argument("--wandb-entity", default=None, help="W&B entity (username or team); defaults to your default entity")
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    args = p.parse_args()

    TRAIN_OUTPUT_FOLDER = args.output_dir
    if not TRAIN_OUTPUT_FOLDER:
        TRAIN_OUTPUT_FOLDER = _find_latest_stgcn_training_dir()
        if not TRAIN_OUTPUT_FOLDER:
            print(
                "No suitable * _training folder found under output/ "
                "(need config.json + best_model.pth). "
                "Run training.py first or pass --output-dir."
            )
            return
        print(f"Using checkpoint directory: {TRAIN_OUTPUT_FOLDER}")
    TRAIN_OUTPUT_FOLDER = os.path.abspath(TRAIN_OUTPUT_FOLDER)
    DATA_DIR = os.path.abspath(args.data_dir)
    print(f"Dataset directory: {DATA_DIR}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    test_save_dir = os.path.join(_OUTPUT_ROOT, f"{ts}_test")
    os.makedirs(test_save_dir, exist_ok=True)

    config_path = os.path.join(TRAIN_OUTPUT_FOLDER, "config.json")
    model_path = os.path.join(TRAIN_OUTPUT_FOLDER, "best_model.pth")

    if not os.path.exists(config_path):
        base = os.path.basename(TRAIN_OUTPUT_FOLDER.rstrip(os.sep))
        hint = ""
        if base.endswith("_test"):
            hint = (
                "\n  You pointed at a *test* output folder (..._test). "
                "--output-dir must be a *training* folder (..._training) that "
                "contains config.json and best_model.pth. "
                "See checkpoint_source.txt inside your ..._test folder for the path "
                "that was used, or list output/*_training."
            )
        else:
            hint = (
                "\n  Expected config.json in that folder. "
                "Use a directory from training.py, e.g. output/YYYYMMDD_HHMMSS_training."
            )
        print(f"Config not found: {config_path}{hint}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_wandb = not args.no_wandb

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{ts}_test",
            job_type="test",
            config={**cfg, "checkpoint_dir": TRAIN_OUTPUT_FOLDER},
        )

    test_dataset = KarateDataset(DATA_DIR, class_names=cfg["classes"], mode="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = STGCN(
        num_classes=len(cfg["classes"]),
        in_channels=cfg["in_channels"],
        num_nodes=cfg["num_nodes"],
        interaction_mode=_interaction_mode_from_cfg(cfg),
    ).to(device)
    model.load_state_dict(_load_state_dict(model_path, device))
    model.eval()

    all_preds, all_labels = [], []
    print("Evaluating Test Set...")
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            out = model(data)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels, all_preds, target_names=cfg["classes"], digits=4
    )
    report_dict = classification_report(
        all_labels, all_preds, target_names=cfg["classes"], output_dict=True
    )
    print("\n" + "=" * 30 + "\nTEST RESULTS\n" + "=" * 30)
    print(report)

    with open(
        os.path.join(test_save_dir, "checkpoint_source.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(TRAIN_OUTPUT_FOLDER)

    with open(os.path.join(test_save_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    _save_confusion_matrix(
        cm, cfg["classes"],
        save_path=os.path.join(test_save_dir, "test_cm.png"),
        txt_path=os.path.join(test_save_dir, "test_cm.txt"),
    )

    if use_wandb:
        per_class = {
            f"test/per_class/{cls}/f1": report_dict[cls]["f1-score"]
            for cls in cfg["classes"]
            if cls in report_dict
        }
        wandb.log({
            "test/accuracy": report_dict["accuracy"],
            "test/macro_f1": report_dict["macro avg"]["f1-score"],
            "test/weighted_f1": report_dict["weighted avg"]["f1-score"],
            "test/confusion_matrix": wandb.Image(os.path.join(test_save_dir, "test_cm.png")),
            **per_class,
        })
        wandb.finish()

    print(f"Results saved to {test_save_dir}")


if __name__ == "__main__":
    test_model()
