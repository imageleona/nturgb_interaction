# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Data preprocessing and training pipeline for the **NTU RGB+D 120** interaction subset (actions A050–A060, A106–A120). Converts raw `.skeleton` files into `.npy` arrays, then into training-ready JSON files, and trains an ST-GCN model for skeleton-based action recognition.

## Pipeline

```
.skeleton files → txt2npy.py → npy_output/*.npy → npy2json_training.py → json_output/ → training.py → output/
```

### Step 1 — Raw conversion

```bash
python txt2npy.py
```

Configure at the top of `txt2npy.py`:

```python
save_npy_path = '/path/to/npy_output/'     # where .npy files are written
load_txt_path = '/path/to/skeleton/root/'  # root dir containing A050/, A051/, ... subdirs
missing_file_path = './ntu_rgb120_missings.txt'
step_ranges = list(range(0, 100))          # S-index range; use non-overlapping ranges for parallelism
```

### Step 2 — Training JSON conversion

```bash
python npy2json_training.py
```

Configure `load_npy_path` and `save_json_path` at the top. Applies an 80/20 train/test split per class (seeded for reproducibility). Output:

```
json_output/
  A050/
    train/A050_train.json
    test/A050_test.json
  ...
```

### Step 3 — Training

```bash
conda activate GNN
python training.py
```

Key arguments: `--epochs`, `--batch-size`, `--lr`, `--num-frames`, `--no-interaction`. Saves checkpoints and plots to `output/<timestamp>_training/`.

### Step 4 — Testing

```bash
python test.py
```

Automatically loads the latest `output/*_training/` checkpoint. Override with `--output-dir`. Saves confusion matrix and classification report to `output/<timestamp>_test/`.

## Script Roles

| Script | Role |
|---|---|
| `txt2npy.py` | Parses `.skeleton` text files, saves each sample as a `.npy` dict |
| `npy2json_training.py` | Converts `.npy` to training-ready JSON with 80/20 train/test split per class |
| `npy2json.py` | Utility — dumps `.npy` dicts to generic JSON as-is, for inspection |
| `dataset.py` | PyTorch `Dataset` (`KarateDataset`); loads JSON, pads/resamples to fixed frame count |
| `model.py` | ST-GCN model with NTU-25 skeleton graph; supports `full` / `hand_cross` / `none` interaction modes |
| `training.py` | Training loop with Adam, MultiStepLR, best-model checkpointing, CSV + plot logging |
| `test.py` | Evaluation on test split; outputs confusion matrix and per-class classification report |
| `ntu_rgb120_missings.txt` | Static list of 535 samples with incomplete tracking; skipped during conversion |

## Model Architecture (`model.py`)

**`Graph`** builds a 3-partition spatial adjacency matrix (self / centripetal / centrifugal) over 50 nodes (25 NTU joints × 2 persons). Cross-person edges are controlled by `interaction_mode`:
- `"full"` — every P1 joint ↔ every P2 joint
- `"hand_cross"` — wrists (joints 6, 10) linked to all joints on the other person
- `"none"` — two independent skeletons

**`STGCN`** stacks 4 ST-GCN blocks (64→64→128→128 channels, stride-2 at layer 3), followed by global average pooling and a 1×1 conv classifier. Input shape: `(N, 2, T, 50)`.

## Dataset (`dataset.py`)

`KarateDataset` loads all `*_{mode}.json` files under `data_dir` recursively. Label is taken from `content["index"]` (1-based) if present, otherwise matched from the class name in the filename. Each sequence is padded or linspace-resampled to `DEFAULT_NUM_FRAMES = 150`. Output tensor shape: `(2, num_frames, 50)`.

Class names constant: `NTU_INTERACTION_CLASS_NAMES` — 26 classes in sorted order (A050–A060, A106–A120). Always pass this as `class_names` when instantiating `KarateDataset` for NTU data.

`extract_two_person_xy_flat` handles `detected_joints == 50` (NTU, added branch) as well as 34 and 54 (legacy LIMU formats).

## Data Formats

### `.npy` dict

Loaded with `np.load(f, allow_pickle=True).item()`:

```python
{
  'file_name': str,               # original .skeleton filename
  'njoints':   25,
  'nbodys':    [int, ...],        # body count per non-empty frame
  'skel_body0': (nframes, 25, 3), # 3D world XYZ
  'rgb_body0':  (nframes, 25, 2), # RGB image projection (x, y)  ← used for training
  'depth_body0':(nframes, 25, 2), # depth image projection
  # body1 keys present only when a 2nd person appears
}
```

Frame counts vary from 17 to 214 (median 57).

### Training JSON

```json
{ "index": 1, "data": [ [[100 floats per frame], ...], ...sequences ] }
```

Each frame row is **100 floats** — person 1 block then person 2 block:
```
indices  0–49 : person 1 — [j0_x, j0_y, j1_x, j1_y, ..., j24_x, j24_y]
indices 50–99 : person 2 — [j0_x, j0_y, j1_x, j1_y, ..., j24_x, j24_y]
```

To extract joint `k` of person `p` (0-indexed): `row[(p * 50) + k * 2]` = x, `+1` = y.

### NTU-25 joint order

| 0 Base of spine | 1 Mid spine | 2 Neck | 3 Head |
|---|---|---|---|
| 4 Left shoulder | 5 Left elbow | 6 Left wrist | 7 Left hand |
| 8 Right shoulder | 9 Right elbow | 10 Right wrist | 11 Right hand |
| 12 Left hip | 13 Left knee | 14 Left ankle | 15 Left foot |
| 16 Right hip | 17 Right knee | 18 Right ankle | 19 Right foot |
| 20 Upper spine | 21 Left hand tip | 22 Left thumb | 23 Right hand tip | 24 Right thumb |

## File Naming Convention

```
S###C###P###R###A###.skeleton
```
= Scene / Camera / Person / Replication / **Action**. Action class extracted from `name[16:20]` (e.g. `"A050"`).

## Data on Disk

~24,800 `.skeleton` files in `A050/`–`A060/` and `A106/`–`A120/` subdirectories (180–500 KB each). `npy_output/` holds one `.npy` per sample (flat directory). `json_output/` holds one train + one test JSON per class.
