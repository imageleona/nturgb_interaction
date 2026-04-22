#!/usr/bin/env python
# coding=utf-8

'''
transform the skeleton data in NTU RGB+D dataset into the numpy arrays for a more efficient data loading
'''

import numpy as np
import os
import sys
import glob

save_npy_path = 'C:/Users/_s2111724/Documents/nturgb_interaction/npy_output/'
load_txt_path = 'C:/Users/_s2111724/Documents/nturgb_interaction/'
missing_file_path = './ntu_rgb120_missings.txt'
step_ranges = list(range(0, 100))  # sequence S-index range; set narrower for parallel runs


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

def _load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line not in missing_files:
                missing_files[line] = True
    return missing_files

def _read_skeleton(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    nframe = int(datas[0].strip())
    bodymat = dict()
    bodymat['file_name'] = os.path.basename(file_path)
    bodymat['nbodys'] = []
    bodymat['njoints'] = njoints
    for body in range(max_body):
        if save_skelxyz:
            bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))

    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor].strip())
        if bodycount == 0:
            continue
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)

            bodyinfo = datas[cursor].strip().split(' ')
            cursor += 1

            njoints = int(datas[cursor].strip())
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor].strip().split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame, joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame, joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame, joint] = jointinfo[5:7]

    if not bodymat['nbodys']:
        return None

    for each in range(max_body):
        if each >= max(bodymat['nbodys']):
            if save_skelxyz:
                del bodymat['skel_body{}'.format(each)]
            if save_rgbxy:
                del bodymat['rgb_body{}'.format(each)]
            if save_depthxy:
                del bodymat['depth_body{}'.format(each)]
    return bodymat


if __name__ == '__main__':
    os.makedirs(save_npy_path, exist_ok=True)
    missing_files = _load_missing_file(missing_file_path)
    datalist = glob.glob(os.path.join(load_txt_path, '**', '*.skeleton'), recursive=True)
    alread_exist = os.listdir(save_npy_path)
    alread_exist_dict = dict(zip(alread_exist, len(alread_exist) * [True]))

    for ind, each in enumerate(datalist):
        each_name = os.path.basename(each)
        _print_toolbar(ind * 1.0 / len(datalist),
                       '({:>5}/{:<5})'.format(ind + 1, len(datalist)))
        S = int(each_name[1:4])
        if S not in step_ranges:
            continue
        if each_name + '.npy' in alread_exist_dict:
            print('file already existed !')
            continue
        if each_name[:20] in missing_files:
            print('file missing')
            continue
        print(each_name)
        mat = _read_skeleton(each)
        if mat is None:
            print('skipping {} — all frames empty'.format(each_name))
            continue
        save_path = save_npy_path + '{}.npy'.format(each_name)
        np.save(save_path, mat)

    _end_toolbar()
