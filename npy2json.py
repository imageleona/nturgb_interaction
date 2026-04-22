#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os
import sys
import json
import glob

load_npy_path = 'C:/Users/_s2111724/Documents/nturgb_interaction/npy_output/'
save_json_path = 'C:/Users/_s2111724/Documents/nturgb_interaction/json_output/'


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


if __name__ == '__main__':
    os.makedirs(save_json_path, exist_ok=True)
    datalist = glob.glob(os.path.join(load_npy_path, '*.npy'))

    for ind, each in enumerate(datalist):
        each_name = os.path.basename(each)
        _print_toolbar(ind * 1.0 / len(datalist),
                       '({:>5}/{:<5})'.format(ind + 1, len(datalist)))

        mat = np.load(each, allow_pickle=True).item()

        json_mat = {}
        for key, val in mat.items():
            if isinstance(val, np.ndarray):
                json_mat[key] = val.tolist()
            else:
                json_mat[key] = val

        save_path = os.path.join(save_json_path, each_name.replace('.npy', '.json'))
        with open(save_path, 'w') as f:
            json.dump(json_mat, f)

    _end_toolbar()
    print('Done. {} files converted.'.format(len(datalist)))
