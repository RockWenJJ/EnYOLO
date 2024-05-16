import torch
import torch.nn as nn
from copy import deepcopy
from torch import Tensor
from typing import Dict, List, Tuple, Union, Optional
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import OptSampleList, SampleList
from mmengine.dist import get_world_size
from mmengine.logging import print_log

from mmengine.optim import OptimWrapper
from enyolo.registry import MODELS
from enyolo.models.losses import CoralLoss

@MODELS.register_module()
class EnYOLO(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 decode_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 res_data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True
                 ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')
        
        self.decode_head = MODELS.build(decode_head)
        # get various data_preprocessor, each is designed for one loss
        self.res_data_preprocessor = MODELS.build(res_data_preprocessor)
        self.coral_loss = CoralLoss()
    
    def det_data_res_preprocessor(self, det_data):
        '''Get restored data for detection'''
        det_data = self.data_preprocessor(det_data, training=True)
        data_res = self._run_forward(det_data, mode='tensor', cat='res')
        det_data_ori = deepcopy(det_data)
        det_data_res = det_data
        
        det_data_ori['inputs'] = det_data['inputs'].detach()
        det_data_res['inputs'] = data_res
        
        return det_data_ori, det_data_res
    
        
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper,
                   stage: str = 'burn_in') -> Dict[str, Tensor]:
        '''
        stage should be 'burn_in/mutual_learn/domain_adapt
        '''
        if stage.lower() == 'burn_in':
            with optim_wrapper.optim_context(self):
                det_data, res_data = data[0], data[1]
                
                det_data = self.data_preprocessor(det_data, training=True)
                losses = self._run_forward(det_data, mode='loss', cat='det')
                
                res_data = self.res_data_preprocessor(res_data, training=True)
                res_losses = self._run_forward(res_data, mode='loss', cat='res')
                
                losses.update(res_losses)
        elif stage.lower() =='mutual_learn':
            with optim_wrapper.optim_context(self):
                det_data, res_data, det_data_ml = data[0], data[1], data[2]
    
                det_data = self.data_preprocessor(det_data, training=True)
                losses = self._run_forward(det_data, mode='loss', cat='det')
    
                res_data = self.res_data_preprocessor(res_data, training=True)
                res_losses = self._run_forward(res_data, mode='loss', cat='res')
                losses.update(res_losses)
    
                # restore real-world degraded images and compute unsupervised loss
                det_data_ori, det_data_res = self.det_data_res_preprocessor(det_data_ml)
                uns_losses = self._run_forward(det_data_res, mode='loss', cat='uns')
                losses.update(uns_losses)
                
                # compute detection loss for restored images
                detres_losses_tmp = self._run_forward(det_data_res, mode='loss', cat='det')
                detres_losses = {}
                for k, v in detres_losses_tmp.items():
                    detres_losses[k+'_res'] = v
                losses.update(detres_losses)
                
        elif stage.lower() == 'domain_adapt':
            with optim_wrapper.optim_context(self):
                det_data, res_data, det_data_ml = data[0], data[1], data[2]
        
                det_data = self.data_preprocessor(det_data, training=True)
                losses = self._run_forward(det_data, mode='loss', cat='det')
        
                res_data = self.res_data_preprocessor(res_data, training=True)
                res_losses = self._run_forward(res_data, mode='loss', cat='res')
                losses.update(res_losses)

                # restore real-world degraded images and compute unsupervised loss
                det_data_ori, det_data_res = self.det_data_res_preprocessor(det_data_ml)
                uns_losses = self._run_forward(det_data_res, mode='loss', cat='uns')
                losses.update(uns_losses)

                # compute detection loss for restored images
                # det_data_ori, det_data_res = self.det_data_res_preprocessor(det_data_ml, data_ml_res)
                detres_losses_tmp = self._run_forward(det_data_res, mode='loss', cat='det')
                detres_losses = {}
                for k, v in detres_losses_tmp.items():
                    detres_losses[k + '_res'] = v
                losses.update(detres_losses)
                
                # compute domain adaptation loss
                da_data = dict(inputs=det_data_ori['inputs'], inputs_res=det_data_res['inputs'].detach())
                da_losses = self._run_forward(da_data, mode='loss', cat='da')
                losses.update(da_losses)
                
        else:
            raise RuntimeError(f"{stage} is not supported for EnYOLO.")

        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars
    
    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str,
                     cat: str = 'det') -> Union[Dict[str, torch.Tensor], list]:
        if isinstance(data, dict):
            results = self(**data, mode=mode, cat=cat)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode, cat=cat)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        
        return results
    
    def forward(self,
                inputs: torch.Tensor,
                mode: str = 'tensor',
                cat: str = 'det',
                *args, **kwargs):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, cat, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, cat, **kwargs)
        elif mode == 'tensor':
            return self._forward(inputs, cat, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
            
    
    def loss(self, batch_inputs: Tensor,
             cat: str = 'det', **kwargs) -> Union[dict, list]:
        '''cat should be det/res/uns/da'''
        if cat == 'det':
            batch_data_samples = kwargs['data_samples']
            x = self.extract_feat(batch_inputs)
            detr_losses = self.bbox_head.loss(x, batch_data_samples)
            return detr_losses
        elif cat == 'res':
            batch_targets = kwargs['targets']
            x = self.backbone(batch_inputs)
            res_losses = self.decode_head.loss(x, batch_inputs, batch_targets)
            return res_losses
        elif cat == 'uns':
            # batch_res = self._run_forward(batch_inputs, mode='tensor', cat='res')
            # compute un-supervised loss
            data_ml_res_tmp = torch.reshape(batch_inputs, list(batch_inputs.shape[:-2]) + [-1])
            uns_losses = torch.mean((torch.mean(data_ml_res_tmp, -1) - 0.5) ** 2) * 100
            return dict(unsup_loss=uns_losses)
        elif cat == 'da':
            batch_inputs_res = kwargs['inputs_res']
            feats_ori = self.backbone(batch_inputs)
            feats_res = self.backbone(batch_inputs_res)
            losses_da = []
            for feat_ori, feat_res in zip(feats_ori, feats_res):
                mse_loss = nn.MSELoss()(feat_ori, feat_res.detach())
                coral_loss = self.coral_loss(feat_ori, feat_res.detach())
                losses_da.append(coral_loss + mse_loss)
                # l1_loss = nn.L1Loss()(feat_ori, feat_res.detach())
                # losses_da.append(l1_loss)
            da_losses = sum(losses_da)
            return dict(da_loss=da_losses)
        else:
            raise RuntimeError(f'{cat} loss func is not implemented in EnYOLO')
            
    def predict(self, batch_inputs: Tensor,
                cat: str = 'det', **kwargs) -> Tuple[List[Tensor]]:
        ''' cat should be either det or res'''
        if cat == 'det':
            batch_data_samples = kwargs['data_samples']
            x = self.extract_feat(batch_inputs)
            results_list = self.bbox_head.predict(
                x, batch_data_samples, rescale=True)
            batch_data_samples = self.add_pred_to_datasample(
                batch_data_samples, results_list)
            return batch_data_samples
        elif cat == 'res':
            results = self._forward(batch_inputs, cat='res')
            # TODO: when do prediction, results should be parsed into [0., 1.] or [0, 255]
            mean, std = self.res_data_preprocessor.mean, self.res_data_preprocessor.std
            results = results * std + mean
            return results
        else:
            raise RuntimeError(f'{cat} predict func is not implemented in EnYOLO')
    
    def _forward(self, batch_inputs: Tensor,
                cat: str = 'det', **kwargs):
        ''' cat should be either det or res'''
        if cat == 'det':
            x = self.extract_feat(batch_inputs)
            results = self.bbox_head.forward(x)
            return results
        elif cat == 'res':
            x = self.backbone(batch_inputs)
            results = self.decode_head.forward(x, batch_inputs)
            return results
        else:
            raise RuntimeError(f'{cat} _forward func is not implemented in EnYOLO')

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        '''Replace the original extract feat because the backbone feats are
        different are longer than the neck feats.'''
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x[2:])
        return x