import json
from json.tool import main
import os
import cv2
import imagesize
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset
from mmengine.dataset import Compose
from mmengine.logging import print_log

from ..registry import DATASETS


@DATASETS.register_module()
class TestImagesDataset(Dataset):
    def __init__(self,
                 pipeline,
                 ann_file=None,
                 data_root=None,
                 img_suffix='.png',
                 test_mode=False):
        super().__init__()
        self.data_root = data_root
        self.input_prefix = data_root
        self.img_suffix = img_suffix
        self.test_mode = test_mode
        
        if ann_file is not None:
            self.ann_file = os.path.join(data_root, ann_file)
            self.data_infos = self.load_annotations(self.ann_file)
        else:
            # parse images and get data_infos
            imgs = glob(os.path.join(data_root, '*' + img_suffix))
            infos = []
            print_log('Parse image infos for restoration ...')
            for img in tqdm(imgs):
                h, w = imagesize.get(img)
                name = os.path.basename(img)
                info = dict(filename=name, height=h, width=w)
                infos.append(info)
            self.data_infos = infos
            
        self._set_group_flag()
        self.pipeline = Compose(pipeline)
    
    def load_annotations(self, ann_file):
        return json.load(open(ann_file, 'r'))
    
    def _set_group_flag(self):
        '''Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        '''
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
    
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        '''Get train/test data from pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            data (dict): Train/Test data
        '''
        if self.test_mode:
            return self._prepare_test_img(idx)
        
        while True:
            data = self._prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    def _prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def _prepare_test_img(self, idx):
        return self._prepare_train_img(idx)
    
    def pre_pipeline(self, results):
        '''Prepare results dict for pipeline.'''
        results['input_prefix'] = self.input_prefix
        return results