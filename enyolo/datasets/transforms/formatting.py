import numpy as np
from mmcv.transforms import to_tensor, BaseTransform
from mmdet.datasets.transforms import PackDetInputs as MMDET_PackDetInputs
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData, PixelData

from enyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackResInputs(BaseTransform):
    """Pack the input data for restoration.
    """
    def __init__(self,
                 meta_keys=('file_name', 'img_shape', 'ori_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys
        
    def transform(self, results: dict) -> dict:
        """Method to pack the input data for restoration."""
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()
            packed_results['inputs'] = img

        if 'target' in results:
            target = results['target']
            if len(target.shape) < 3:
                target = np.expand_dims(target, -1)
            if not target.flags.c_contiguous:
                target = np.ascontiguousarray(target.transpose(2, 0, 1))
                target = to_tensor(target)
            else:
                target = to_tensor(target).permute(2, 0, 1).contiguous()
            packed_results['targets'] = target
        
        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                                   f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]
        
        packed_results['metainfo'] = img_meta
        
        return packed_results