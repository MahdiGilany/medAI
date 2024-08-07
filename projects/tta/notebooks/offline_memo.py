import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import sys
sys.path.append(os.getenv("PROJECT_PATH"))


import torch
import torch.nn as nn
import typing as tp
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
import logging
import wandb

import medAI
from medAI.utils.setup import BasicExperiment, BasicExperimentConfig

from utils.metrics import MetricCalculator

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import timm

from copy import copy, deepcopy
import pandas as pd

from datasets.datasets import ExactNCT2013RFImagePatches
from medAI.datasets.nct2013 import (
    KFoldCohortSelectionOptions,
    LeaveOneCenterOutCohortSelectionOptions, 
    PatchOptions
)

for LEAVE_OUT in ["UVA"]: # "JH", "PCC", "CRCEO", "PMCC", "UVA"
    print("Leave out", LEAVE_OUT)

    ## Data Finetuning
    ###### No support dataset ######

    from baseline_experiment import BaselineConfig
    config = BaselineConfig(cohort_selection_config=LeaveOneCenterOutCohortSelectionOptions(leave_out=f"{LEAVE_OUT}"))

    from torchvision.transforms import v2 as T
    from torchvision.tv_tensors import Image as TVImage
    class Transform:
        def __init__(selfT, augment=False):
            selfT.augment = augment
            selfT.size = (256, 256)
            # Augmentation
            selfT.transform = T.Compose([
                # T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ])  
            # selfT.transform = T.Compose([
            #     T.Resize(selfT.size, antialias=True)
            # ])  
        
        def __call__(selfT, item):
            patch = item.pop("patch")
            patch = copy(patch)
            patch = (patch - patch.min()) / (patch.max() - patch.min()) \
                if config.instance_norm else patch
            patch = TVImage(patch)
            patch = T.Resize(selfT.size, antialias=True)(patch).float()
            
            label = torch.tensor(item["grade"] != "Benign").long()
            
            if selfT.augment:
                patch_augs = torch.stack([selfT.transform(patch) for _ in range(5)], dim=0)
                return patch_augs, patch, label, item
            
            return -1, patch, label, item

    # val_ds = ExactNCT2013RFImagePatches(
    #     split="val",
    #     transform=Transform(augment=True),
    #     cohort_selection_options=config.cohort_selection_config,
    #     patch_options=config.patch_config,
    #     debug=config.debug,
    # )
    
    if isinstance(config.cohort_selection_config, LeaveOneCenterOutCohortSelectionOptions):
        if config.cohort_selection_config.leave_out == "UVA":
            config.cohort_selection_config.benign_to_cancer_ratio = 5.0 
    
    test_ds = ExactNCT2013RFImagePatches(
        split="test",
        transform=Transform(augment=True),
        cohort_selection_options=config.cohort_selection_config,
        patch_options=config.patch_config,
        debug=config.debug,
    )


    # val_loader = DataLoader(
    #     val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    # )

    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    )


    ## Model
    from vicreg_pretrain_experiment import TimmFeatureExtractorWrapper
    from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
    fe_config = config.model_config
    # Create the model
    model: nn.Module = timm.create_model(
        fe_config.model_name,
        num_classes=fe_config.num_classes,
        in_chans=1,
        features_only=fe_config.features_only,
        norm_layer=lambda channels: nn.GroupNorm(
                        num_groups=fe_config.num_groups,
                        num_channels=channels
                        ))

    CHECkPOINT_PATH = os.path.join(f'/ssd005/projects/exactvu_pca/checkpoint_store/Mahdi/baseline_gn_crtd3ratio_loco/baseline_gn_crtd3ratio_loco_{LEAVE_OUT}/', 'best_model.ckpt')
    # CHECkPOINT_PATH = os.path.join(f'/fs01/home/abbasgln/codes/medAI/projects/tta/logs/tta/baseline_gn_avgprob_3ratio_loco/baseline_gn_avgprob_3ratio_loco_{LEAVE_OUT}/', 'best_model.ckpt')
    # CHECkPOINT_PATH = os.path.join(f'/ssd005/projects/exactvu_pca/checkpoint_store/Mahdi/baseline_gn_3ratio_loco-noexcltrn/', 'best_model.ckpt')


    model.load_state_dict(torch.load(CHECkPOINT_PATH)['model'])
    model.eval()
    model.cuda()
    
    
    ## MEMO
    loader = test_loader
    enable_memo = True

    from memo_experiment import batched_marginal_entropy
    metric_calculator = MetricCalculator()
    desc = "test"

    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(tqdm(loader, desc=desc)):
        batch = deepcopy(batch)
        images_augs, images, labels, meta_data = batch
        images_augs = images_augs.cuda()
        images = images.cuda()
        labels = labels.cuda()
        
        batch_size, aug_size= images_augs.shape[0], images_augs.shape[1]

        # Adapt to test
        _images_augs = images_augs.reshape(-1, *images_augs.shape[2:]).cuda()
        adaptation_model = deepcopy(model)
        adaptation_model.eval()
        if enable_memo:
            optimizer = optim.SGD(adaptation_model.parameters(), lr=1e-3)
            
            for j in range(1):
                outputs = adaptation_model(_images_augs).reshape(batch_size, aug_size, -1)  
                loss, logits = batched_marginal_entropy(outputs)
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
        
        # Evaluate
        logits = adaptation_model(images)
        loss = criterion(logits, labels)
                        
        # Update metrics   
        metric_calculator.update(
            batch_meta_data = meta_data,
            probs = nn.functional.softmax(logits, dim=-1).detach().cpu(),
            labels = labels.detach().cpu(),
        )
    
    ## Find metrics
    # Log metrics every epoch
    metric_calculator.avg_core_probs_first = True
    metrics = metric_calculator.get_metrics()

    # Update best score
    (best_score_updated,best_score) = metric_calculator.update_best_score(metrics, desc)

    best_score_updated = copy(best_score_updated)
    best_score = copy(best_score)
            
    # Log metrics
    metrics_dict = {
        f"{desc}/{key}": value for key, value in metrics.items()
        }

    print(metrics_dict)
    
    ## Log with wandb
    import wandb
    # group=f"offline_tent_5it_e-3lr_gn_3ratio_loco"
    group=f"offline_memo_3aug_e-3lr_gn_3ratio_loco"
    # group=f"offline_baseline_5e-4lr_gn_avgprob_3ratio_loco"
    # group=f"offline_newBaseline_1e-4lr2ep_gn_avgprob_3ratio_loco"
    # group=f"baseline_gn_3ratio_loco-noexcltrn"
    
    # Save logits
    save_dir = f"/ssd005/projects/exactvu_pca/checkpoint_store/Mahdi/saved_logits/{group}/{LEAVE_OUT}"
    os.makedirs(save_dir, exist_ok=True) if not os.path.exists(save_dir) else None
    torch.save({
        "core_id_probs": metric_calculator.core_id_probs,
        "core_id_labels": metric_calculator.core_id_labels},
        os.path.join(save_dir, "core_id_probs_labels.pth")
        )
    
    name= group + f"_{LEAVE_OUT}"
    wandb.init(project="tta", entity="mahdigilany", name=name, group=group)
    # os.environ["WANDB_MODE"] = "enabled"
    metrics_dict.update({"epoch": 0})
    wandb.log(
        metrics_dict,
        )
    wandb.finish()
    
    del test_ds, test_loader, loader, model