import math
import os
from dotenv import load_dotenv
# Loading environment variables
load_dotenv()

import sys
sys.path.append(os.getenv("PROJECT_PATH"))


import torch
import torch.nn as nn
import torch.nn.functional as F
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



for LEAVE_OUT in ["PCC"]: # "JH", "PCC", "PMCC", "UVA", "CRCEO"
    print("Leave out", LEAVE_OUT)
    
    ## Data Finetuning
    ###### No support dataset ######

    from ensemble_experiment import EnsembleConfig
    config = EnsembleConfig(cohort_selection_config=LeaveOneCenterOutCohortSelectionOptions(leave_out=f"{LEAVE_OUT}"),
    )

    from baseline_experiment import BaselineConfig
    from torchvision.transforms import v2 as T
    from torchvision.tv_tensors import Image as TVImage

    class Transform:
        def __init__(selfT, augment=False):
            selfT.augment = augment
            selfT.size = (256, 256)
            # Augmentation
            # selfT.transform = T.Compose([
            #     T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            #     T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.5),
            #     T.RandomHorizontalFlip(p=0.5),
            #     T.RandomVerticalFlip(p=0.5),
            # ])  
            selfT.transform = T.Compose([
                T.Resize(selfT.size, antialias=True)
            ])  
                
        def __call__(selfT, item):
            patch = item.pop("patch")
            patch = copy(patch)
            patch = (patch - patch.min()) / (patch.max() - patch.min()) \
                if config.instance_norm else patch
            patch = TVImage(patch)
            patch = T.Resize(selfT.size, antialias=True)(patch).float()
            
            label = torch.tensor(item["grade"] != "Benign").long()
            
            if selfT.augment:
                patch_augs = torch.stack([selfT.transform(patch) for _ in range(1)], dim=0)
                return patch_augs, patch, label, item
            
            return -1, patch, label, item

    
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

    batch_size = config.batch_size
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # val_ds = ExactNCT2013RFImagePatches(
    #     split="val",
    #     transform=Transform(augment=False),
    #     cohort_selection_options=config.cohort_selection_config,
    #     patch_options=config.patch_config,
    #     debug=config.debug,
    # )
    
    # val_loader = DataLoader(
    #     val_ds, batch_size=config.batch_size, shuffle=True, num_workers=4
    # )
    
    
    ## Model
    from baseline_experiment import FeatureExtractorConfig
    from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
    from MI_ensemble_experiment import TimmFeatureExtractorWrapper

    fe_config = FeatureExtractorConfig()
    # Create the model
    models: tp.List[nn.Module] = [timm.create_model(
        fe_config.model_name,
        num_classes=fe_config.num_classes,
        in_chans=1,
        features_only=True,
        norm_layer=lambda channels: nn.GroupNorm(
                        num_groups=fe_config.num_groups,
                        num_channels=channels
                        )) for _ in range(5)]
    
    global_pools = [SelectAdaptivePool2d(pool_type='avg',flatten=True,input_fmt='NCHW').cuda() 
                for _ in range(config.num_ensembles)]
    
    fe_models = [nn.Sequential(TimmFeatureExtractorWrapper(model), global_pool) 
                for model, global_pool in zip(models, global_pools)]
    
    linears = [nn.Linear(512, config.model_config.num_classes).cuda()
               for _ in range(config.num_ensembles)]
        
    CHECkPOINT_PATH = os.path.join(f'/ssd005/projects/exactvu_pca/checkpoint_store/Mahdi/MIensemble_10mi_5mdls_3ratio_gn_loco2/MIensemble_10mi_5mdls_3ratio_gn_loco2_{LEAVE_OUT}/', 'best_model.ckpt')

    state = torch.load(CHECkPOINT_PATH)
    [model.load_state_dict(state["list_fe_models"][i]) for i, model in enumerate(fe_models)]
    [linear.load_state_dict(state["list_linears"][i]) for i, linear in enumerate(linears)]
    
    list_models = [nn.Sequential(fe_model, linear) for fe_model, linear in zip(fe_models, linears)]
    [model.eval() for model in list_models]
    [model.cuda() for model in list_models]
    
    
    '''
    ## Temp Scaling
    loader = val_loader

    metric_calculator = MetricCalculator()
    desc = "val"


    temp = torch.tensor(1.0).cuda().requires_grad_(True)
    beta = torch.tensor(0.0).cuda().requires_grad_(True)


    params = [temp, beta]
    _optimizer = optim.Adam(params, lr=1e-3)

    for epoch in range(1):
        metric_calculator.reset()
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            images_augs, images, labels, meta_data = batch
            images = images.cuda()
            labels = labels.cuda()
            

            # Evaluate
            with torch.no_grad():
                stacked_logits = torch.stack([model(images) for model in list_models])
            scaled_stacked_logits = stacked_logits/ temp + beta
            losses = [nn.CrossEntropyLoss()(
                scaled_stacked_logits[i, ...],
                labels
                ) for i in range(5)
            ]
            
            # optimize
            _optimizer.zero_grad()
            sum(losses).backward()
            _optimizer.step()
                        
            # Update metrics   
            metric_calculator.update(
                batch_meta_data = meta_data,
                probs = nn.functional.softmax(scaled_stacked_logits, dim=-1).mean(dim=0).detach().cpu(), # Take mean over ensembles
                labels = labels.detach().cpu(),
            )
    print("temp beta", temp, beta)
    '''
    
    
    temp = 1.0
    beta = 0.0
    if LEAVE_OUT == "JH":
    #     temp = 1.6793
    #     beta = -1.0168
        temp = 0.9253
        beta = -1.0273
    elif LEAVE_OUT == "PCC":
    #     temp = 1.5950
    #     beta = -0.8514
        temp = 1.0075
        beta = -0.8614
    elif LEAVE_OUT == "PMCC":
    #     temp = 0.6312
    #     beta = -1.0017
        temp = 0.9020
        beta = -1.0609
    elif LEAVE_OUT == "UVA":
    #     temp = 0.9333
    #     beta = -0.7474
        temp = 1.6528
        beta = -0.6192
    elif LEAVE_OUT == "CRCEO":
    #     temp = 1.2787
    #     beta = -0.8716
        temp = 0.8515
        beta = -0.8461
        
    temp = torch.tensor(temp).cuda()
    beta = torch.tensor(beta).cuda()
    
    
    ## Test-time Adaptation
    from memo_experiment import batched_marginal_entropy
    from MI_ensemble_experiment import mutual_info_loss
    
    loader = test_loader
    enable_uplabel = True
    temp_scale = False
    unc_thr = 1.

    metric_calculator = MetricCalculator()
    desc = "test"


    for i, batch in enumerate(tqdm(loader, desc=desc)):
        images_augs, images, labels, meta_data = batch
        images_augs = images_augs.cuda()
        images = images.cuda()
        labels = labels.cuda()
        
        adaptation_model_list = [deepcopy(model) for model in list_models] 
        [model.train() for model in adaptation_model_list]
        # no reset
        # adaptation_model_list = list_models 
        
        if enable_uplabel:
            batch_size, aug_size= images_augs.shape[0], images_augs.shape[1]
            params = []
            for model in adaptation_model_list:
                params.append({'params': model.parameters()})
            optimizer = optim.SGD(params, lr=1e-3)
            _images_augs = images_augs.reshape(-1, *images_augs.shape[2:]).cuda()
            
            # Adapt to test 
            for j in range(1):
                # Forward pass
                stacked_logits = torch.stack([model(_images_augs).reshape(batch_size, aug_size, -1) for model in adaptation_model_list]) # (n_models, batch_size, aug_size, num_classes)
                if temp_scale:
                    stacked_logits = stacked_logits / temp + beta
                    
                list_losses = []
                
                ###### Losses ######
                # Core Cross-Entropy
                # avg_marginal_probs = F.softmax(stacked_logits, dim=-1).mean(dim=0).mean(dim=0).mean(dim=0).expand(batch_size,-1).detach() # (1, num_classes)
                # marginal_probs = F.softmax(stacked_logits, dim=-1).mean(dim=0).mean(dim=1) # (batch_size, num_classes)
                # loss = nn.CrossEntropyLoss()(marginal_probs, avg_marginal_probs)
                
                ## Combined Cross-Entropy
                # marginal_probs = F.softmax(stacked_logits, dim=-1).mean(dim=0).mean(dim=1) # (batch_size, num_classes)
                # loss = nn.CrossEntropyLoss()(marginal_probs, labels)
            
                ## calculating marginal entropy based on all models predictions
                perm_logits = torch.permute(stacked_logits, (1, 0, 2, 3)) # (batch_size, n_models, aug_size, num_classes)
                all_logits = torch.reshape(perm_logits, (batch_size, -1, perm_logits.shape[-1])) # (batch_size, n_models*aug_size, num_classes)
                entropy_loss, avg_logits = batched_marginal_entropy(all_logits) 
                # SAR
                margin = unc_thr*math.log(2)
                idx = torch.where(entropy_loss < margin)
                breakpoint()
                avg_logits = avg_logits[idx]
                pseudo_labels = avg_logits.argmax(dim=-1).detach() # (batch_size)
                loss = nn.CrossEntropyLoss()(avg_logits, pseudo_labels)
                # mutual information loss
                # mi_coeff = 0.0
                # MI_loss = mutual_info_loss(F.softmax(stacked_logits.reshape(-1,batch_size*aug_size, 2), dim=-1))
                # loss = entropy_loss.mean() + mi_coeff * MI_loss
                
                ## calculating marginal entropy for each model separately
                # for k in range(5):
                #     loss, logits = batched_marginal_entropy(stacked_logits[k,...])
                #     ## Combined Cross-Entropy
                #     # loss = nn.CrossEntropyLoss()(marginal_probs[k,...], avg_marginal_probs)
                #     list_losses.append(loss.mean())
                #####################
                
                optimizer.zero_grad()          
                # sum(list_losses).backward()
                loss.backward()
                optimizer.step()
        
        # Evaluate
        [model.eval() for model in adaptation_model_list]
        logits = torch.stack([model(images) for model in adaptation_model_list])
        if temp_scale:
            logits = logits / temp + beta
        losses = [nn.CrossEntropyLoss()(
            logits[i, ...],
            labels
            ) for i in range(5)
        ]
                        
        # Update metrics   
        metric_calculator.update(
            batch_meta_data = meta_data,
            probs = F.softmax(logits, dim=-1).mean(dim=0).detach().cpu(), # Take mean over ensembles
            labels = labels.detach().cpu(),
        )
    
    
    # Log metrics every epoch
    metrics = metric_calculator.get_metrics(acc_threshold=0.3)

    # Update best score
    (best_score_updated,best_score) = metric_calculator.update_best_score(metrics, desc)

    best_score_updated = copy(best_score_updated)
    best_score = copy(best_score)
            
    # Log metrics
    metrics_dict = {
        f"{desc}/{key}": value for key, value in metrics.items()
        }
    
    print(metrics_dict)
    print(metric_calculator.get_metrics(acc_threshold=0.2))
    print(metric_calculator.get_metrics(acc_threshold=0.4))
    print(metric_calculator.get_metrics(acc_threshold=0.6))
    print(metric_calculator.get_metrics(acc_threshold=0.7))
    
        
    ## Log with wandb
    import wandb
    group=f"offline_MIEnsmUPLabel_1unc-thr_e-3lr_gn_3ratio_loco"
    
    
    print(group)
    name= group + f"_{LEAVE_OUT}"
    wandb.init(project="tta", entity="mahdigilany", name=name, group=group)
    # os.environ["WANDB_MODE"] = "enabled"
    metrics_dict.update({"epoch": 0})
    wandb.log(
        metrics_dict,
        )
    
    
    wandb.finish()
    # del val_ds, test_ds, val_loader, test_loader, loader, list_models
    del test_ds,  test_loader, loader, list_models