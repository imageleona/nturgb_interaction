"""
create_subsets.py

Produces stratified random subsets of the NTU RGB+D training JSONs for
dataset-size ablation studies. Each subset mirrors the json_output/ tree so
that training.py can consume it directly via --data-dir.

Usage:
    python create_subsets.py
    python create_subsets.py --src-dir json_output --dst-dir json_subsets --seed 42
    python create_subsets.py --ratios 0.1 0.25 0.5 1.0
"""

import argparse
import glob
import json
import math
import os
import random
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_SEED = 42


def _ratio_to_dir(ratio: float) -> str:
    return str(int(round(ratio * 100)))


def _discover_train_files(src_dir: str) -> list:
    all_files = glob.glob(os.path.join(src_dir, "**", "*.json"), recursive=True)
    train_files = [
        os.path.abspath(p)
        for p in all_files
        if os.path.basename(p).lower().endswith("train.json")
    ]
    return sorted(train_files)


def _load_class_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  WARNING: could not read {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(content.get("data"), list) or len(content["data"]) == 0:
        print(f"  WARNING: {path} has no valid 'data' list — skipping.", file=sys.stderr)
        return None
    return content


def build_subsets(src_dir: str, dst_dir: str, ratios: list, seed: int) -> None:
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    train_files = _discover_train_files(src_dir)
    if not train_files:
        print(f"ERROR: no *_train.json files found under {src_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(train_files)} class train file(s) under {src_dir}")
    print(f"Ratios : {ratios}")
    print(f"Output : {dst_dir}\n")

    # Pre-shuffle each class once.  All ratios then slice the same prefix of
    # that permutation so the 20% set is always a strict subset of the 40% set.
    class_data = {}
    for path in train_files:
        content = _load_class_json(path)
        if content is None:
            continue
        class_index = int(content.get("index", 0))
        rng = random.Random(seed + class_index)
        indices = list(range(len(content["data"])))
        rng.shuffle(indices)
        class_data[path] = {"content": content, "shuffled": indices}

    if not class_data:
        print("ERROR: no valid class files could be loaded.", file=sys.stderr)
        sys.exit(1)

    for ratio in ratios:
        dir_name = _ratio_to_dir(ratio)
        ratio_root = os.path.join(dst_dir, dir_name)
        total_seqs = 0

        print(f"--- {int(round(ratio * 100))}% -> {ratio_root} ---")

        for src_path, info in class_data.items():
            content = info["content"]
            shuffled = info["shuffled"]
            seqs = content["data"]

            n_take = max(1, math.ceil(len(seqs) * ratio))
            chosen = [seqs[i] for i in shuffled[:n_take]]

            rel_path = os.path.relpath(src_path, src_dir)
            out_path = os.path.join(ratio_root, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"index": content["index"], "data": chosen}, f)

            total_seqs += n_take
            print(f"  {rel_path}: {len(seqs)} -> {n_take}")

        print(f"  Total: {total_seqs} sequences\n")

    print("Done.")


def main():
    p = argparse.ArgumentParser(
        description="Create stratified training-data subsets for NTU RGB+D ablations."
    )
    p.add_argument("--src-dir", default=os.path.join(_SCRIPT_DIR, "json_output"),
                   help="Root directory with *_train.json files (default: json_output/)")
    p.add_argument("--dst-dir", default=os.path.join(_SCRIPT_DIR, "json_subsets"),
                   help="Output root directory (default: json_subsets/)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Base random seed (default: {DEFAULT_SEED})")
    p.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
                   help=f"Sampling ratios (default: {DEFAULT_RATIOS})")
    args = p.parse_args()

    for r in args.ratios:
        if not (0.0 < r <= 1.0):
            print(f"ERROR: ratio {r} is out of range (0, 1]", file=sys.stderr)
            sys.exit(1)

    build_subsets(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        ratios=sorted(args.ratios),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
