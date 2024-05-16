import random
from typing import List, Optional, Tuple, Union

import torch
import math
import torch.nn.functional as F
from mmdet.models import BatchSyncRandomResize
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine import MessageHub, is_list_of, is_seq_of
from mmengine.structures import BaseDataElement
from mmengine.model import ImgDataPreprocessor, stack_batch
from torch import Tensor

from enyolo.registry import MODELS

@MODELS.register_module()
class PairedResDataPreprocessor(ImgDataPreprocessor):
    def forward(self, data: dict, training: bool = False,
                keys=['inputs', 'targets']) -> Union[dict, list]:
        '''Perform normalization, padding and bgr2rgb conversion based on
        `BaseDataPreprocessor`.'''
        
        data = self.cast_data(data)
        for k in keys:
            if k in data.keys():
                _batch_inputs = data[k]
                if is_seq_of(_batch_inputs, torch.Tensor):
                    batch_inputs = []
                    for _batch_input in _batch_inputs:
                        # channel transform
                        if self._channel_conversion:
                            _batch_input = _batch_input[[2, 1, 0], ...]
                        # Convert to float after channel conversion to ensure
                        # efficiency
                        _batch_input = _batch_input.float()
                        # Normalization.
                        if self._enable_normalize:
                            if self.mean.shape[0] == 3:
                                assert _batch_input.dim(
                                ) == 3 and _batch_input.shape[0] == 3, (
                                    'If the mean has 3 values, the input tensor '
                                    'should in shape of (3, H, W), but got the tensor '
                                    f'with shape {_batch_input.shape}')
                            _batch_input = (_batch_input - self.mean) / self.std
                        batch_inputs.append(_batch_input)
                    # Pad and stack Tensor.
                    batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                               self.pad_value)
                # Process data with `default_collate`.
                elif isinstance(_batch_inputs, torch.Tensor):
                    assert _batch_inputs.dim() == 4, (
                        'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                        'or a list of tensor, but got a tensor with shape: '
                        f'{_batch_inputs.shape}')
                    if self._channel_conversion:
                        _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
                    # Convert to float after channel conversion to ensure
                    # efficiency
                    _batch_inputs = _batch_inputs.float()
                    if self._enable_normalize:
                        _batch_inputs = (_batch_inputs - self.mean) / self.std
                    h, w = _batch_inputs.shape[2:]
                    target_h = math.ceil(
                        h / self.pad_size_divisor) * self.pad_size_divisor
                    target_w = math.ceil(
                        w / self.pad_size_divisor) * self.pad_size_divisor
                    pad_h = target_h - h
                    pad_w = target_w - w
                    batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                         'constant', self.pad_value)
                else:
                    raise TypeError('Output of `cast_data` should be a dict of '
                                    'list/tuple with inputs and data_samples, '
                                    f'but got {type(data)}： {data}')
                
                # if not training:
                #     data[k] = batch_inputs
                #     data.setdefault('data_samples', None)
                data[k] = batch_inputs
                data.setdefault('data_samples', None)
                
        return data
                
            
        
        