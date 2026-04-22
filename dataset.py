import os
import json
import glob
import warnings
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

INPUT_CHANNELS = 2
DEFAULT_NUM_FRAMES = 150

NTU_INTERACTION_CLASS_NAMES: List[str] = [
    "A050", "A051", "A052", "A053", "A054", "A055", "A056", "A057", "A058", "A059", "A060",
    "A106", "A107", "A108", "A109", "A110", "A111", "A112", "A113", "A114", "A115", "A116",
    "A117", "A118", "A119", "A120",
]


def split_sequences_from_raw_data(raw_data) -> List:
    if not isinstance(raw_data, list) or len(raw_data) == 0:
        return []
    if (
        isinstance(raw_data[0], list)
        and len(raw_data[0]) > 0
        and isinstance(raw_data[0][0], list)
    ):
        return raw_data
    return [raw_data]


def extract_two_person_xy_flat(seq_array: np.ndarray) -> Optional[np.ndarray]:
    """(frames, feat_dim) -> (frames, N_joints, 2) or None."""
    if seq_array.ndim != 2:
        return None
    frames, feat_dim = seq_array.shape
    if feat_dim % 2 != 0:
        return None
    detected_joints = feat_dim // INPUT_CHANNELS
    tmp = seq_array.reshape(frames, detected_joints, INPUT_CHANNELS)
    if detected_joints == 50:  # NTU 25 joints x 2 persons
        p1 = tmp[:, 0:25, :]
        p2 = tmp[:, 25:50, :]
        return np.concatenate([p1, p2], axis=1)
    if detected_joints >= 54:
        p1 = tmp[:, 0:17, :]
        p2 = tmp[:, 27:44, :]
        return np.concatenate([p1, p2], axis=1)
    if detected_joints >= 34:
        p1 = tmp[:, 0:17, :]
        p2 = tmp[:, 17:34, :]
        return np.concatenate([p1, p2], axis=1)
    return None


def pad_resample_time(arr: np.ndarray, num_frames: int) -> np.ndarray:
    """Pad or linspace-resample along time (dim 0). `arr` is (T, ...)."""
    t = arr.shape[0]
    nf = int(num_frames)
    if t < nf:
        pad_shape = (nf - t,) + tuple(arr.shape[1:])
        return np.vstack((arr.astype(np.float32), np.zeros(pad_shape, dtype=np.float32)))
    indices = np.linspace(0, t - 1, nf).astype(int)
    return arr[indices].astype(np.float32)


class KarateDataset(Dataset):
    def __init__(self, data_dir, class_names, mode="train",
                 do_center=False, do_scale=False,
                 num_frames: Optional[int] = None):
        self.data_list = []
        self.labels = []
        self.class_names = class_names
        self.mode = mode
        self.do_center = do_center
        self.do_scale = do_scale
        self.num_frames = int(num_frames) if num_frames is not None else DEFAULT_NUM_FRAMES
        self.num_joints = 50
        self.in_channels = INPUT_CHANNELS
        self._load_errors = 0
        self._load_data(data_dir)
        if len(self.data_list) > 0:
            self.in_channels = int(self.data_list[0].shape[0])

    def _normalize_sequence(self, sequence):
        if not (isinstance(sequence, np.ndarray) and sequence.ndim == 2):
            return sequence
        frames, dim = sequence.shape
        num_joints = dim // INPUT_CHANNELS
        seq = sequence.reshape(frames, num_joints, INPUT_CHANNELS)
        if self.do_center and num_joints > 0:
            center_xy = seq[:, 0:1, :]  # NTU joint 0: base of spine
            seq = seq - center_xy
        if self.do_scale:
            masked = seq.copy()
            masked[masked == 0] = np.nan
            scale = np.nanmax(np.abs(masked))
            if not np.isfinite(scale) or scale < 1e-6:
                scale = 1.0
            seq = seq / scale
        return seq.reshape(frames, dim)

    def _load_data(self, data_dir):
        print(f"\n--- Loading {self.mode} dataset ---")
        all_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
        target_suffix = f"{self.mode}.json"
        valid_samples = 0

        for json_file in all_files:
            filename = os.path.basename(json_file)
            if not filename.lower().endswith(target_suffix.lower()):
                continue

            label_idx = -1
            for idx, name in enumerate(self.class_names):
                if name.lower() in filename.lower():
                    label_idx = idx
                    break

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                if "index" in content:
                    idx_val = int(content["index"]) - 1
                    if 0 <= idx_val < len(self.class_names):
                        label_idx = idx_val
                if label_idx == -1:
                    continue

                for seq in split_sequences_from_raw_data(content.get("data", [])):
                    seq_array = np.array(seq, dtype=np.float32)
                    if seq_array.ndim != 2:
                        continue
                    seq_array = np.nan_to_num(seq_array, nan=0.0, posinf=0.0, neginf=0.0)
                    seq_array = self._normalize_sequence(seq_array)
                    filtered = extract_two_person_xy_flat(seq_array)
                    if filtered is None:
                        continue
                    final_seq = pad_resample_time(filtered, self.num_frames).transpose(2, 0, 1)
                    self.data_list.append(final_seq)
                    self.labels.append(label_idx)
                    valid_samples += 1
            except Exception as e:
                self._load_errors += 1
                warnings.warn(f"Skip {json_file}: {e}", UserWarning)

        if self._load_errors:
            print(f"Warning: {self._load_errors} file(s) failed to load.")
        print(f"Loaded {valid_samples} samples. Each sample has {self.num_joints} nodes ({self.num_joints // 2} per person).")
        if valid_samples > 0:
            shp = self.data_list[0].shape
            print(f"Sample tensor shape: {tuple(shp)} (channels={shp[0]}, time_steps={self.num_frames}, num_nodes={self.num_joints})")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data_list[idx]).float()
        return x, torch.tensor(int(self.labels[idx]), dtype=torch.long)
