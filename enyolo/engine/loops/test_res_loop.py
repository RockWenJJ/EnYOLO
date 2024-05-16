import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from mmengine.registry import LOOPS
from mmengine.runner import BaseLoop

from copy import deepcopy
from itertools import cycle

@LOOPS.register_module()
class TestResLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 fp16: bool = False):
        super().__init__(runner, dataloader)
        
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch test."""
        self.runner.model.eval()
        results = []
        for idx, data_batch in enumerate(self.dataloader):
            results.append(self.run_iter(idx, data_batch))
            
        return results

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.
        """
        
        data = self.runner.model.res_data_preprocessor(data_batch, training=False)
        data_res = self.runner.model._run_forward(data, mode='tensor', cat='res')

        out = (torch.clip(data_res, 0, 1) * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        out_img = Image.fromarray(out)
        img_name = data_batch['img_names'][0]
        
        result = dict(image=out_img, img_name=img_name)
        
        return result
        