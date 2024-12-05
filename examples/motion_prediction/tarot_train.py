import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import os


def tarot_selection(candidate_loader, target_loader, model, config='unitraj/tarot_config.yaml'):
    import yaml
    import glob
    from tarot.wfd_estimator import WFDEstimator
    from tarot.data_selector import DataSelector
    from tarot.utils import qualitative_analysis, set_dataloader_params
    from utils.visualization import visualize_batch_data 

    from tqdm import tqdm
    def transform_batch(batch):
        inp_dict = batch['input_dict']
        batch = [x.to(device) for x in inp_dict.values() if type(x) == torch.Tensor]
        keys = [k for k in inp_dict.keys() if type(inp_dict[k]) == torch.Tensor]
        batch.append(keys)
        return batch
    print("Current working directory:", os.getcwd())
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    
    _candidate_loader = set_dataloader_params(candidate_loader,'shuffle', False)
    _target_loader = set_dataloader_params(target_loader,'shuffle', False)

    device = cfg['device']
    exp_name = cfg['exp_name']

    ckpt_usage = cfg['ckpt_usage']
    ckpt_files = glob.glob( os.path.join(cfg['ckpt_dir'], '**', '*.ckpt'), recursive=True)
    ckpt_files = sorted(ckpt_files, key=os.path.getmtime)
    ckpt_files = ckpt_files[-ckpt_usage:]
    print(f'use {ckpt_files} to get score')

    ckpts = [torch.load(ckpt, map_location=device)['state_dict'] for ckpt in ckpt_files]
    feature_save_dir = cfg['feature_save_dir']
    model.load_state_dict(ckpts[-1])
    model = model.eval()
    model = model.to(device)
    wfd_estimator = WFDEstimator(model=model,
                                 task='motion_prediction',
                                 proj_dim=cfg['projection_dimension'],
                                 candidate_set_size=len(candidate_loader.dataset),target_set_size=len(target_loader.dataset), device=device, load_from_save_dir=True,
                                 save_dir=feature_save_dir, use_half_precision=False)

    for model_id, ckpt in enumerate(tqdm(ckpts)):
        wfd_estimator.load_checkpoint(ckpt, model_id=model_id)
        print('collecting grads for candidate set')
        for batch in tqdm(_candidate_loader):
            batch = transform_batch(batch)
            wfd_estimator.collect_grads(batch=batch, num_samples=batch[0].shape[0])
        print('collecting grads for target set')
        for batch in tqdm(_target_loader):
            batch = transform_batch(batch)
            wfd_estimator.collect_grads(batch=batch, num_samples=batch[0].shape[0],is_target=True)

    scores,candidate_features, target_features = wfd_estimator.get_wfd()

    data_selector = DataSelector(cfg)

    selected_index, weight = data_selector.select_data(scores, candidate_features, target_features)
    
    qualitative_analysis(exp_name, selected_index, weight, scores, candidate_features, target_features, _candidate_loader, _target_loader, visualize_batch_data)

    train_loader = data_selector.update_dataloader_with_weights(candidate_loader,target_loader,selected_index, weight)



    return train_loader


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices), 1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',  # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        save_top_k=1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        dirpath=f'./unitraj_ckpt/{cfg.exp_name}'
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers,shuffle=True, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=True, drop_last=False,
        collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=None if cfg.debug else WandbLogger(project="unitraj", name=cfg.exp_name, id=cfg.exp_name),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="auto" if cfg.debug else "ddp",
        callbacks=call_backs
    )

    cfg['val_data_path'] = cfg["target_data_path"]
    target_set = build_dataset(cfg, val=True)
    target_loader = DataLoader(
        target_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)
    train_loader = tarot_selection(train_loader, target_loader, model)

    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = os.path.join('./unitraj', cfg.exp_name, '**', '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    train()
