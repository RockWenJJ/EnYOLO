import math
import os
from copy import deepcopy
from typing import List, Sequence, Tuple, Union, Optional

import mmcv
import mmengine.fileio as fileio
import numpy as np

from mmengine.registry import TRANSFORMS
from mmcv.transforms import LoadImageFromFile, Resize, RandomResize, RandomFlip
from mmdet.datasets.transforms import RandomCrop


@TRANSFORMS.register_module()
class LoadTestImagesFromFile(LoadImageFromFile):
    def transform(self, results: dict) -> Optional[dict]:
        """Funcctions to load both synthetic and clear images"""
        
        img_name = results['img_info']['filename']
        input_prefix = results['input_prefix']
        
        input_filename = os.path.join(input_prefix, img_name)
        
        try:
            if self.file_client_args is not None:
                input_file_client = fileio.FileClient.infer_client(
                    self.file_client_args, input_filename)
                input_img_bytes = input_file_client.get(input_filename)
            else:
                input_img_bytes = fileio.get(
                    input_filename, backend_args=self.backend_args)
            input_img = mmcv.imfrombytes(
                input_img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        
        assert input_img is not None, f'failed to load image: {input_filename}'
        if self.to_float32:
            input_img = input_img.astype(np.float32)
        
        results['img'] = input_img
        results['img_shape'] = input_img.shape[:2]
        results['ori_shape'] = input_img.shape[:2]
        results['file_name'] = img_name
        
        return results
    
    
@TRANSFORMS.register_module()
class ResizeTestImage(Resize):
    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        return results
