#!/usr/bin/env python
# coding=utf-8

"""
Visualize learned edge importance weights from a trained ST-GCN model.

For each of the 9 ST-GCN layers produces:
  - Left panel  : skeleton drawing with edges colored by within-person importance
  - Right panel : 25x25 heatmap of cross-person edge importance (P1 joints x P2 joints)

Usage:
    python visualize_edges.py                        # uses latest output/*_training
    python visualize_edges.py --output-dir output/20240101_120000_training
"""

import os
import json
import argparse
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from model import STGCN

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_ROOT = os.path.join(_SCRIPT_DIR, "output")

# 2D canonical positions for NTU-25 joints (x right, y up)
JOINT_POS = np.array([
    [ 0.00,  0.00],  #  0 base of spine
    [ 0.00,  0.30],  #  1 mid spine
    [ 0.00,  0.85],  #  2 neck
    [ 0.00,  1.05],  #  3 head
    [-0.40,  0.70],  #  4 left shoulder
    [-0.70,  0.45],  #  5 left elbow
    [-0.90,  0.20],  #  6 left wrist
    [-1.05,  0.05],  #  7 left hand
    [ 0.40,  0.70],  #  8 right shoulder
    [ 0.70,  0.45],  #  9 right elbow
    [ 0.90,  0.20],  # 10 right wrist
    [ 1.05,  0.05],  # 11 right hand
    [-0.20, -0.10],  # 12 left hip
    [-0.22, -0.50],  # 13 left knee
    [-0.22, -0.90],  # 14 left ankle
    [-0.25, -1.10],  # 15 left foot
    [ 0.20, -0.10],  # 16 right hip
    [ 0.22, -0.50],  # 17 right knee
    [ 0.22, -0.90],  # 18 right ankle
    [ 0.25, -1.10],  # 19 right foot
    [ 0.00,  0.60],  # 20 upper spine
    [-1.15, -0.05],  # 21 left hand tip
    [-1.00,  0.15],  # 22 left thumb
    [ 1.15, -0.05],  # 23 right hand tip
    [ 1.00,  0.15],  # 24 right thumb
])

NTU_EDGES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]

NTU_JOINT_NAMES = [
    "base spine", "mid spine", "neck", "head",
    "L shoulder", "L elbow", "L wrist", "L hand",
    "R shoulder", "R elbow", "R wrist", "R hand",
    "L hip", "L knee", "L ankle", "L foot",
    "R hip", "R knee", "R ankle", "R foot",
    "upper spine", "L hand tip", "L thumb", "R hand tip", "R thumb",
]


def _find_latest_training_dir() -> Optional[str]:
    if not os.path.isdir(_OUTPUT_ROOT):
        return None
    best_path, best_mtime = None, -1.0
    for name in os.listdir(_OUTPUT_ROOT):
        if not name.endswith("_training"):
            continue
        path = os.path.join(_OUTPUT_ROOT, name)
        if not (os.path.isfile(os.path.join(path, "config.json")) and
                os.path.isfile(os.path.join(path, "best_model.pth"))):
            continue
        m = os.path.getmtime(path)
        if m > best_mtime:
            best_mtime, best_path = m, path
    return best_path


def _get_layer_effective_weights(gcn_module) -> np.ndarray:
    """Sum A * edge_importance across the 3 partitions -> (50, 50)."""
    A   = gcn_module.A.detach().cpu().numpy()                # (3, 50, 50)
    imp = gcn_module.edge_importance.detach().cpu().numpy()  # (3, 50, 50)
    return (A * imp).sum(axis=0)                             # (50, 50)


def _draw_skeleton(ax, joint_pos, edge_weights, cmap, norm, person_label):
    """Draw one skeleton. edge_weights is a dict {(i,j): float}."""
    for (i, j), w in edge_weights.items():
        x = [joint_pos[i, 0], joint_pos[j, 0]]
        y = [joint_pos[i, 1], joint_pos[j, 1]]
        ax.plot(x, y, color=cmap(norm(w)), linewidth=2.5 + 3.0 * norm(w), solid_capstyle="round")
    ax.scatter(joint_pos[:, 0], joint_pos[:, 1], c="white", s=30, zorder=5, edgecolors="grey", linewidths=0.5)
    ax.text(0.5, -0.04, person_label, transform=ax.transAxes, ha="center", fontsize=8, color="grey")


def visualize(output_dir: str, save_path: str):
    config_path = os.path.join(output_dir, "config.json")
    model_path  = os.path.join(output_dir, "best_model.pth")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    interaction_mode = cfg.get("interaction_mode", "full")
    model = STGCN(
        num_classes=len(cfg["classes"]),
        in_channels=cfg["in_channels"],
        num_nodes=cfg["num_nodes"],
        interaction_mode=interaction_mode,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    in_ch = cfg["in_channels"]
    layers = [
        model.layer1.gcn, model.layer2.gcn, model.layer3.gcn,
        model.layer4.gcn, model.layer5.gcn, model.layer6.gcn,
        model.layer7.gcn, model.layer8.gcn, model.layer9.gcn,
    ]
    layer_names = [
        f"Layer 1 ({in_ch}→64)",
        "Layer 2 (64→64)",
        "Layer 3 (64→64)",
        "Layer 4 (64→128, stride 2)",
        "Layer 5 (128→128)",
        "Layer 6 (128→128)",
        "Layer 7 (128→256, stride 2)",
        "Layer 8 (256→256)",
        "Layer 9 (256→256)",
    ]

    cmap = cm.get_cmap("YlOrRd")

    # person offsets for side-by-side skeleton drawing
    p1_offset = np.array([-1.6, 0.0])
    p2_offset = np.array([ 1.6, 0.0])

    fig, axes = plt.subplots(9, 2, figsize=(14, 46))
    fig.suptitle("Learned Edge Importance per ST-GCN Layer", fontsize=13, y=0.995)

    for row, (gcn, lname) in enumerate(zip(layers, layer_names)):
        eff = _get_layer_effective_weights(gcn)  # (50, 50)

        # --- within-person edge weights ---
        within_p1, within_p2 = {}, {}
        for (i, j) in NTU_EDGES:
            w_p1 = (eff[i, j] + eff[j, i]) / 2
            w_p2 = (eff[i+25, j+25] + eff[j+25, i+25]) / 2
            within_p1[(i, j)] = w_p1
            within_p2[(i, j)] = w_p2

        all_within = list(within_p1.values()) + list(within_p2.values())
        norm_within = Normalize(vmin=min(all_within), vmax=max(all_within))

        ax_skel = axes[row, 0]
        ax_skel.set_facecolor("#1a1a2e")
        ax_skel.set_aspect("equal")
        ax_skel.axis("off")
        ax_skel.set_title(lname, fontsize=9, pad=4)

        _draw_skeleton(ax_skel, JOINT_POS + p1_offset, within_p1, cmap, norm_within, "Person 1")
        _draw_skeleton(ax_skel, JOINT_POS + p2_offset, within_p2, cmap, norm_within, "Person 2")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_within)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_skel, fraction=0.02, pad=0.02, label="importance")

        # --- cross-person heatmap (P1 joints x P2 joints) ---
        cross = (eff[:25, 25:] + eff[25:, :25].T) / 2  # average both directions

        ax_heat = axes[row, 1]
        im = ax_heat.imshow(cross, cmap="YlOrRd", aspect="auto")
        ax_heat.set_title(f"{lname} — cross-person", fontsize=9, pad=4)
        ax_heat.set_xlabel("Person 2 joint", fontsize=7)
        ax_heat.set_ylabel("Person 1 joint", fontsize=7)
        ticks = list(range(25))
        ax_heat.set_xticks(ticks)
        ax_heat.set_xticklabels(NTU_JOINT_NAMES, rotation=90, fontsize=5)
        ax_heat.set_yticks(ticks)
        ax_heat.set_yticklabels(NTU_JOINT_NAMES, fontsize=5)
        fig.colorbar(im, ax=ax_heat, fraction=0.02, pad=0.02, label="importance")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=None,
                   help="Training run folder (default: latest output/*_training)")
    p.add_argument("--save-path", default=None,
                   help="Where to save the figure (default: inside --output-dir)")
    args = p.parse_args()

    output_dir = args.output_dir or _find_latest_training_dir()
    if not output_dir:
        print("No *_training folder found. Run training.py first or pass --output-dir.")
        return

    output_dir = os.path.abspath(output_dir)
    save_path = args.save_path or os.path.join(output_dir, "edge_importance.png")
    visualize(output_dir, save_path)


if __name__ == "__main__":
    main()
