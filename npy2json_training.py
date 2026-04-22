#!/usr/bin/env python
# coding=utf-8

'''
Convert .npy skeleton files to training-ready JSON files.

Output structure:
    json_output/
        A050/
            train/A050_train.json
            test/A050_test.json
        A051/
            ...

JSON format per file:
    {
        "index": <1-based class index>,
        "data": [
            [[p1_j0_x, p1_j0_y, ..., p1_j24_y, p2_j0_x, ..., p2_j24_y], ...],  <- sequence 1
            ...                                                                    <- sequence N
        ]
    }
Each frame row is 100 floats: 25 joints x 2 coords for person 1, then person 2 (rgb_body).

NOTE: dataset.py's extract_two_person_xy_flat must handle detected_joints == 50.
Add this branch before the >= 34 check:
    if detected_joints == 50:
        p1 = tmp[:, 0:25, :]
        p2 = tmp[:, 25:50, :]
        return np.concatenate([p1, p2], axis=1)
And set self.num_joints = 50 in KarateDataset.__init__.
'''

import numpy as np
import os
import json
import glob
import random
import sys
from collections import defaultdict

load_npy_path = 'C:/Users/_s2111724/Documents/nturgb_interaction/npy_output/'
save_json_path = 'C:/Users/_s2111724/Documents/nturgb_interaction/json_output/'
train_ratio = 0.8
seed = 42


toolbar_width = 50

def _print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def _end_toolbar():
    sys.stdout.write('\n')


def mat_to_frame_list(mat):
    '''
    Returns a list of frame rows (100 floats each), or None if fewer than 2 bodies.
    Layout: [p1_j0_x, p1_j0_y, p1_j1_x, ..., p1_j24_y, p2_j0_x, ..., p2_j24_y]
    '''
    if 'rgb_body0' not in mat or 'rgb_body1' not in mat:
        return None
    p1 = mat['rgb_body0']  # (nframes, 25, 2)
    p2 = mat['rgb_body1']  # (nframes, 25, 2)
    combined = np.concatenate([p1, p2], axis=1)  # (nframes, 50, 2)
    combined = combined.reshape(combined.shape[0], -1)  # (nframes, 100)
    return combined.tolist()


if __name__ == '__main__':
    random.seed(seed)

    all_files = glob.glob(os.path.join(load_npy_path, '*.npy'))
    if not all_files:
        print('No .npy files found in', load_npy_path)
        sys.exit(1)

    # Group files by action class extracted from filename
    # Filename format: S001C001P001R001A050.skeleton.npy
    # Action class at indices 16:20 -> 'A050'
    class_files = defaultdict(list)
    for f in all_files:
        name = os.path.basename(f)
        action = name[16:20]
        class_files[action].append(f)

    class_names = sorted(class_files.keys())
    print('Found {} action classes: {}'.format(len(class_names), class_names))

    total_train = total_test = total_skipped = 0

    for cls_idx, action in enumerate(class_names):
        files = class_files[action]
        random.shuffle(files)

        split_point = int(len(files) * train_ratio)
        splits = {
            'train': files[:split_point],
            'test':  files[split_point:],
        }

        for split_name, split_files in splits.items():
            sequences = []
            for f in split_files:
                mat = np.load(f, allow_pickle=True).item()
                frames = mat_to_frame_list(mat)
                if frames is None:
                    total_skipped += 1
                    continue
                sequences.append(frames)

            save_dir = os.path.join(save_json_path, action, split_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, '{}_{}.json'.format(action, split_name))
            with open(save_path, 'w') as fh:
                json.dump({'index': cls_idx + 1, 'data': sequences}, fh)

            if split_name == 'train':
                total_train += len(sequences)
            else:
                total_test += len(sequences)

        _print_toolbar(
            (cls_idx + 1) / len(class_names),
            '({:>3}/{:<3}) '.format(cls_idx + 1, len(class_names))
        )

    _end_toolbar()
    print('Done. train={}, test={}, skipped={} (single-person).'.format(
        total_train, total_test, total_skipped
    ))
