

def is_metainfo_lower(cfg):
    """Determine whether the custom metainfo fields are all lowercase."""

    def judge_keys(dataloader_cfg):
        while 'dataset' in dataloader_cfg:
            dataloader_cfg = dataloader_cfg['dataset']
        if 'metainfo' in dataloader_cfg:
            all_keys = dataloader_cfg['metainfo'].keys()
            all_is_lower = all([str(k).islower() for k in all_keys])
            assert all_is_lower, f'The keys in dataset metainfo must be all lowercase, but got {all_keys}. '
            
    judge_keys(cfg.get('train_dataloader', {}))
    judge_keys(cfg.get('val_dataloader', {}))
    judge_keys(cfg.get('test_dataloader', {}))