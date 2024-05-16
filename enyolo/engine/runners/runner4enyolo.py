import bisect
import logging
import time
import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader


from mmengine.registry import RUNNERS, LOOPS
from mmengine.runner import Runner as BaseRunner
from mmengine.runner import BaseLoop, EpochBasedTrainLoop, IterBasedTrainLoop


@RUNNERS.register_module()
class Runner4EnYOLO(BaseRunner):
    def __init__(self, *args, **kwargs):
        train_dataloader_det = kwargs.pop('train_dataloader_det')
        train_dataloader_res = kwargs.pop('train_dataloader_res')
        test_dataloader_res = kwargs.pop('test_dataloader_res')
        test_res_cfg = kwargs.pop('test_res_cfg')
        self._train_dataloader_det = train_dataloader_det
        self._train_dataloader_res = train_dataloader_res
        self._test_dataloader_res = test_dataloader_res
        self._test_res_loop = test_res_cfg
        
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_cfg(cls, cfg):
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            train_dataloader_det=cfg.get('train_dataloader_det'),
            train_dataloader_res=cfg.get('train_dataloader_res'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            test_dataloader_res=cfg.get('test_dataloader_res'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            test_res_cfg=cfg.get('test_res_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )
        return runner
    
    def build_train_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')
    
        loop_cfg = copy.deepcopy(loop)
    
        assert 'type' in loop_cfg, f'Runner4EnYOLO must contain key "type" explicitly'
        assert 'EnYOLO' in loop_cfg['type'], f"{loop_cfg['type']} is not compatible for Runner4EnYOLO"
        
        loop = LOOPS.build(
            loop_cfg,
            default_args=dict(
                runner=self, dataloader=self._train_dataloader,
                dataloader_det=self._train_dataloader_det,
                dataloader_res=self._train_dataloader_res))
        
        return loop
    
    def build_test_res_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build test_res loop.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'test_loop should be a Loop object or dict, but got {loop}')
        
        loop_cfg = copy.deepcopy(loop)
        
        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self,
                    dataloader=self._test_dataloader_res))
        else:
            from enyolo.engine.loops import TestResLoop
            loop = TestResLoop(
                **loop_cfg,
                runner=self,
                dataloader=self._test_dataloader_res)
        
        return loop
    
    @property
    def test_res_loop(self):
        """:obj:`BaseLoop`: A loop to run training."""
        if isinstance(self._test_res_loop, BaseLoop) or self._test_res_loop is None:
            return self._test_res_loop
        else:
            self._test_res_loop = self.build_test_res_loop(self._test_res_loop)
            return self._test_res_loop
    
    
    def test_res(self) -> dict:
        """Launch test for restoration and save restored results.
        """
        if self._test_res_loop is None:
            raise RuntimeError(
                '`self._test_loop` should not be None when calling test '
                'method. Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')

        self._test_res_loop = self.build_test_res_loop(self._test_res_loop)  # type: ignore

        self.call_hook('before_run')

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        results = self.test_res_loop.run()  # type: ignore
        self.call_hook('after_run')
        return results
        