import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmengine.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop, BaseLoop

from copy import deepcopy
from itertools import cycle


@LOOPS.register_module()
class EpochBasedTrainLoop4EnYOLO(EpochBasedTrainLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 dataloader_det: Union[DataLoader, Dict],
                 dataloader_res: Union[DataLoader, Dict],
                 burnin_epochs: int,
                 mutual_epochs: int,
                 max_epochs: int,
                 val_begin: int = 1,
                 val_interval: int = 1,
                 dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader,
                         max_epochs, val_begin, val_interval, dynamic_intervals)
        
        self._burnin_epochs = burnin_epochs
        self._mutual_epochs = mutual_epochs
        
        if isinstance(dataloader_res, dict):
            diff_rank_seed = self.runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.dataloader_det = self.runner.build_dataloader(
                dataloader_det, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
            self.dataloader_res = self.runner.build_dataloader(
                dataloader_res, seed=self.runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.dataloader_det = dataloader_det
            self.dataloader_res = dataloader_res
    
    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook("before_train_epoch")
        self.runner.model.train()
        
        if self._epoch < self._burnin_epochs:
            dataloader_det = iter(self.dataloader)  # dataloader for detection before burinin-stage
            dataloader_res = cycle(iter(self.dataloader_res))  # dataloader for enhancement
            
            iters_per_epoch = len(dataloader_det)
            # do burn-in
            for idx in range(iters_per_epoch):
                # customized run_iter
                data_batch_det = next(dataloader_det)
                data_batch_res = next(dataloader_res)
                data_batch = [data_batch_det, data_batch_res]
                
                # call hook before train_iter
                self.runner.call_hook(
                    'before_train_iter', batch_idx=idx, data_batch=data_batch_det)
                
                # print(self.runner.model)
                log_vars = self.runner.model.train_step(
                    data_batch, optim_wrapper=self.runner.optim_wrapper, stage='burn_in')
                
                
                # call hook after train_iter
                self.runner.call_hook(
                    'after_train_iter',
                    batch_idx=idx,
                    data_batch=data_batch_det,
                    outputs=log_vars)
                
                self._iter += 1
        elif self._epoch < self._mutual_epochs:
            dataloader_det = iter(self.dataloader)
            dataloader_det_ml = iter(self.dataloader_det)  # dataloader for detection for mutual learning
            dataloader_res = cycle(iter(self.dataloader_res))  # dataloader for restoration
            
            iters_per_epoch = len(dataloader_det)
            # do mutual learning
            for idx in range(iters_per_epoch):
                # customized run_iter
                data_batch_det = next(dataloader_det)
                data_batch_res = next(dataloader_res)
                data_batch_det_ml = next(dataloader_det_ml)

                data_batch = [data_batch_det, data_batch_res, data_batch_det_ml]
                
                # call hook before train_iter
                self.runner.call_hook(
                    'before_train_iter', batch_idx=idx, data_batch=data_batch)
                
                log_vars = self.runner.model.train_step(
                    data_batch, optim_wrapper=self.runner.optim_wrapper, stage='mutual_learn')

                # call hook after train_iter
                self.runner.call_hook(
                    'after_train_iter',
                    batch_idx=idx,
                    data_batch=data_batch_det,
                    outputs=log_vars)
                
                self._iter += 1
        else:
            dataloader_det = iter(self.dataloader)
            dataloader_det_ml = iter(self.dataloader_det)  # dataloader for detection for mutual learning
            dataloader_res = cycle(iter(self.dataloader_res))  # dataloader for restoration
    
            iters_per_epoch = len(dataloader_det)
            # do mutual learning
            for idx in range(iters_per_epoch):
                # customized run_iter
                data_batch_det = next(dataloader_det)
                data_batch_det_ml = next(dataloader_det_ml)
                data_batch_res = next(dataloader_res)
        
                data_batch = [data_batch_det, data_batch_res, data_batch_det_ml]
        
                # call hook before train_iter
                self.runner.call_hook(
                    'before_train_iter', batch_idx=idx, data_batch=data_batch)
        
                log_vars = self.runner.model.train_step(
                    data_batch, optim_wrapper=self.runner.optim_wrapper, stage='domain_adapt')
        
                # call hook after train_iter
                self.runner.call_hook(
                    'after_train_iter',
                    batch_idx=idx,
                    data_batch=data_batch_det,
                    outputs=log_vars)
                
                self._iter += 1
        
        self.runner.call_hook('after_train_epoch')
        self._epoch += 1