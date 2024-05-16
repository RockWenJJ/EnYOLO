from typing import List, Sequence

import numpy as np
import torch
from mmengine.dataset import COLLATE_FUNCTIONS
from mmengine.dist import get_dist_info

@COLLATE_FUNCTIONS.register_module()
def paired_image_collate(data_batch: Sequence,
                         use_ms_training: bool = False) -> dict:
    batch_imgs = []
    batch_targets = []
    batch_names = []
    for i in range(len(data_batch)):
        inputs = data_batch[i]['inputs']
        targets = data_batch[i]['targets']
        names = data_batch[i]['metainfo']['file_name']
        batch_imgs.append(inputs)
        batch_targets.append(targets)
        batch_names.append(names)
    
    collated_results = dict(img_names=batch_names)
    if use_ms_training:
        collated_results['inputs'] = batch_imgs
        collated_results['targets'] = batch_targets
    else:
        # get max_h and max_w throughout inptus and targets
        hm, wm = 0, 0
        for img in batch_imgs:
            h, w = img.shape[1:]
            hm = h if h > hm else hm
            wm = w if w > wm else wm
        # make sure hm and wm are divisible to 32
        hm = int(np.ceil(hm / 32) * 32)
        wm = int(np.ceil(wm / 32) * 32)
        
        # pack batch_imgs with the same size
        batch_imgs_tmp = []
        batch_targets_tmp = []
        for img, target in zip(batch_imgs, batch_targets):
            img_tmp = torch.zeros((3, hm, wm), dtype=torch.uint8)
            tag_tmp = torch.zeros((3, hm, wm), dtype=torch.uint8)
            h, w = img.shape[1:]
            hi = (hm - h) // 2
            wi = (wm - w) // 2
            img_tmp[:, hi:hi + h, wi:wi + w] = img
            tag_tmp[:, hi:hi + h, wi:wi + w] = target
            batch_imgs_tmp.append(img_tmp)
            batch_targets_tmp.append(tag_tmp)
        
        collated_results['inputs'] = torch.stack(batch_imgs_tmp, 0)
        collated_results['targets'] = torch.stack(batch_targets_tmp, 0)
    return collated_results


@COLLATE_FUNCTIONS.register_module()
def test_image_collate(data_batch: Sequence,
                         use_ms_training: bool = False) -> dict:
    batch_imgs = []
    batch_names = []
    for i in range(len(data_batch)):
        inputs = data_batch[i]['inputs']
        names = data_batch[i]['metainfo']['file_name']
        batch_imgs.append(inputs)
        batch_names.append(names)
    
    collated_results = dict(img_names=batch_names)
    if use_ms_training:
        collated_results['inputs'] = batch_imgs
    else:
        # get max_h and max_w throughout inptus and targets
        hm, wm = 0, 0
        for img in batch_imgs:
            h, w = img.shape[1:]
            hm = h if h > hm else hm
            wm = w if w > wm else wm
        # make sure hm and wm are divisible to 32
        hm = int(np.ceil(hm / 32) * 32)
        wm = int(np.ceil(wm / 32) * 32)
        
        # pack batch_imgs with the same size
        batch_imgs_tmp = []
        for img in batch_imgs:
            img_tmp = torch.zeros((3, hm, wm), dtype=torch.uint8)
            h, w = img.shape[1:]
            hi = (hm - h) // 2
            wi = (wm - w) // 2
            img_tmp[:, hi:hi + h, wi:wi + w] = img
            batch_imgs_tmp.append(img_tmp)
        
        collated_results['inputs'] = torch.stack(batch_imgs_tmp, 0)
    return collated_results