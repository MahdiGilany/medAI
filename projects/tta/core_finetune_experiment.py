import os
from turtle import back
from dotenv import load_dotenv
from sympy import im
# Loading environment variables
load_dotenv()

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from dataclasses import dataclass
import logging
import wandb

import medAI
from baseline_experiment import BaselineExperiment, BaselineConfig, FeatureExtractorConfig

from utils.metrics import MetricCalculator, CoreMetricCalculator

from timm.optim.optim_factory import create_optimizer

from einops import rearrange, repeat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import timm

from copy import deepcopy, copy
from simple_parsing import subgroups

from datasets.datasets import ExactNCT2013RFImagePatches, ExactNCT2013RFCores
from medAI.datasets.nct2013 import (
    KFoldCohortSelectionOptions,
    LeaveOneCenterOutCohortSelectionOptions, 
    PatchOptions
)

from models.vicreg_module import VICReg
from models.ridge_regression import RidgeRegressor
from timm.layers import create_classifier 
from models.linear_prob import LinearProb
from models.attention import MultiheadAttention as SimpleMultiheadAttention
from itertools import chain
 
 
# # Avoids too many open files error from multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
@dataclass
class AttentionConfig:
    nhead: int = 8
    qk_dim: int = 128
    v_dim: int = 128
    dropout: float= 0.


@dataclass
class FinetunerConfig:
    train_backbone: bool = True
    backbone_lr: float = 1e-4
    head_lr: float = 5e-4
    core_batch_size: int = 16
    attention_config: AttentionConfig = AttentionConfig()
    checkpoint_path_name: str = None 
    

@dataclass
class CoreFinetuneConfig(BaselineConfig):
    """Configuration for the experiment."""
    name: str = " finetune_test_5"
    resume: bool = True
    debug: bool = False
    use_wandb: bool = True
    
    batch_size: int = 1
    epochs: int = 100
    cohort_selection_config: KFoldCohortSelectionOptions | LeaveOneCenterOutCohortSelectionOptions = subgroups(
        {"kfold": KFoldCohortSelectionOptions(fold=0), "loco": LeaveOneCenterOutCohortSelectionOptions(leave_out='PCC')},
        default="loco"
    )
    model_config: FeatureExtractorConfig = FeatureExtractorConfig(features_only=True)
    finetuner_config: FinetunerConfig = FinetunerConfig()


class CoreFinetuneExperiment(BaselineExperiment): 
    config_class = CoreFinetuneConfig
    config: CoreFinetuneConfig

    def __init__(self, config: CoreFinetuneConfig):
        super().__init__(config)
        self.best_val_loss = np.inf
        self.best_score_updated = False
        if self.config.finetuner_config.checkpoint_path_name is None:
            self._checkpoint_path = os.path.join(
                os.getcwd(),
                # f'projects/tta/logs/tta/vicreg_pretrn_2048zdim_gn_loco2/vicreg_pretrn_2048zdim_gn_loco2_{self.config.cohort_selection_config.leave_out}/', 
                f'logs/tta/vicreg_pretrn_2048zdim_gn_loco2/vicreg_pretrn_2048zdim_gn_loco2_{self.config.cohort_selection_config.leave_out}/', 
                'best_model.ckpt'
                )
        else:
            self._checkpoint_path = os.path.join(
                os.getcwd(),
                # f'projects/tta/logs/tta/vicreg_pretrn_2048zdim_gn_loco2/vicreg_pretrn_2048zdim_gn_loco2_{self.config.cohort_selection_config.leave_out}/', 
                f'logs/tta/{self.config.finetuner_config.checkpoint_path_name}/{self.config.finetuner_config.checkpoint_path_name}_{self.config.cohort_selection_config.leave_out}/', 
                'best_model.ckpt'
                )
  
    def setup(self):
        # logging setup
        super(BaselineExperiment, self).setup()
        self.setup_data()
        self.setup_metrics()

        logging.info('Setting up model, optimizer, scheduler')
        self.fe_model, self.attention, self.linear = self.setup_model()
        
        # Optimizer
        if self.config.finetuner_config.train_backbone:
            params = [
                {"params": self.fe_model.parameters(), "lr": self.config.finetuner_config.backbone_lr},
                {"params": self.attention.parameters(), "lr": self.config.finetuner_config.head_lr},
                {"params": self.linear.parameters(), "lr": self.config.finetuner_config.head_lr}
                ]
        else:
            params = [
                {"params": self.attention.parameters(),  "lr": self.config.finetuner_config.head_lr},
                {"params": self.linear.parameters(),  "lr": self.config.finetuner_config.head_lr}
                ]
        
        self.optimizer = optim.Adam(params, weight_decay=1e-6) #lr=self.config.finetuner_config.backbone_lr,
        sched_steps_per_epoch = len(self.train_loader) // self.config.finetuner_config.core_batch_size + 1 
        # sched_steps_per_epoch = len(self.train_loader)  
        self.scheduler = medAI.utils.LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=5 * sched_steps_per_epoch,
            max_epochs=self.config.epochs * sched_steps_per_epoch,
        )
        
        # Setup epoch and best score
        self.epoch = 0 
        
        # Load checkpoint if exists
        if "experiment.ckpt" in os.listdir(self.ckpt_dir) and self.config.resume:
            state = torch.load(os.path.join(self.ckpt_dir, "experiment.ckpt"))
            logging.info(f"Resuming from epoch {state['epoch']}")
        else:
            state = None
            
        if state is not None:
            self.fe_model.load_state_dict(state["fe_model"])
            self.attention.load_state_dict(state["attention"])
            self.linear.load_state_dict(state["linear"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.epoch = state["epoch"]
            self.metric_calculator.initialize_best_score(state["best_score"])
            self.best_score = state["best_score"]
            self.save_states(save_model=False) # Free up model space
            
        # Initialize best score if not resuming
        self.best_score = self.metric_calculator._get_best_score_dict()
            

        logging.info(f"Number of fe parameters: {sum(p.numel() for p in self.fe_model.parameters())}")
        logging.info(f"Number of attention parameters: {sum(p.numel() for p in self.attention.parameters())}")
        logging.info(f"Number of linear parameters: {sum(p.numel() for p in self.linear.parameters())}")

    def setup_data(self):
        from torchvision.transforms import v2 as T

        class Transform:
            def __init__(selfT, augment=False):
                selfT.augment = augment
                selfT.size = (256, 256)
                # Augmentation
                selfT.transform = T.Compose([
                    T.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                    T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                ])  
            
            def __call__(selfT, item):
                patches = item.pop("patches")
                patches = copy(patches)
                patches = (patches - np.min(patches, axis=(-2,-1), keepdims=True)) / (np.min(patches, axis=(-2,-1), keepdims=True) - np.min(patches, axis=(-2,-1), keepdims=True)) \
                    if self.config.instance_norm else patches
                patches = torch.tensor(patches).float()
                patches = T.Resize(selfT.size, antialias=True)(patches).float()
                
                label = torch.tensor(item["grade"] != "Benign").long()
                
                if selfT.augment:
                    patches_augs = torch.stack([selfT.transform(patches) for _ in range(2)], dim=0)
                    return patches_augs, patches, label, item
                
                return -1, patches, label, item


        cohort_selection_options_train = copy(self.config.cohort_selection_config)
        cohort_selection_options_train.min_involvement = self.config.min_involvement_train
        cohort_selection_options_train.benign_to_cancer_ratio = self.config.benign_to_cancer_ratio_train
        cohort_selection_options_train.remove_benign_from_positive_patients = self.config.remove_benign_from_positive_patients_train

        train_ds = ExactNCT2013RFCores(
            split="train",
            transform=Transform(augment=False),
            cohort_selection_options=cohort_selection_options_train,
            patch_options=self.config.patch_config,
            debug=self.config.debug,
        )

        val_ds = ExactNCT2013RFCores(
            split="val",
            transform=Transform(augment=False),
            cohort_selection_options=self.config.cohort_selection_config,
            patch_options=self.config.patch_config,
            debug=self.config.debug,
        )

        if isinstance(self.config.cohort_selection_config, LeaveOneCenterOutCohortSelectionOptions):
            if self.config.cohort_selection_config.leave_out == "UVA":
                self.config.cohort_selection_config.benign_to_cancer_ratio = 5.0 
        test_ds = ExactNCT2013RFCores(
            split="test",
            transform=Transform(augment=True),
            cohort_selection_options=self.config.cohort_selection_config,
            patch_options=self.config.patch_config,
            debug=self.config.debug,
        )


        self.train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True, num_workers=0 #, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=0 #, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config.batch_size, shuffle=False, num_workers=0 #, pin_memory=True
        )

    def setup_model(self):
        from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
        
        fe_model = super().setup_model()
        global_pool = SelectAdaptivePool2d(
            pool_type='avg',
            flatten=True,
            input_fmt='NCHW',
            )
        fe_model = nn.Sequential(TimmFeatureExtractorWrapper(fe_model), global_pool)
        fe_model.load_state_dict(torch.load(self._checkpoint_path)["model"])
        
        attention_config = self.config.finetuner_config.attention_config
        attention = SimpleMultiheadAttention(
            input_dim=512,
            qk_dim=attention_config.qk_dim,
            v_dim=attention_config.v_dim,
            num_heads=attention_config.nhead,
            drop_out=attention_config.dropout
        ).cuda()
        # attention = nn.TransformerEncoderLayer(
        #     d_model=512,
        #     nhead=attention_config.nhead,
        #     dropout=attention_config.dropout,
        #     batch_first=True,
        #     ).cuda()
        
        linear = torch.nn.Sequential(
            # torch.nn.Linear(512, 64),
            torch.nn.Linear(attention_config.nhead*attention_config.v_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.config.model_config.num_classes),
        ).cuda()
        

        return fe_model.cuda(), attention.cuda(), linear.cuda()
    
    def setup_metrics(self):
        self.metric_calculator = CoreMetricCalculator()
    
    def save_states(self, best_model=False, save_model=False):
        torch.save(
            {   
                "fe_model": self.fe_model.state_dict(), # if save_model else None,
                "attention": self.attention.state_dict(), # if save_model else None,
                "linear": self.linear.state_dict() if save_model else None,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
            },
            os.path.join(
                self.ckpt_dir,
                "experiment.ckpt",
            )
        )
        if best_model:
            torch.save(
                {   
                    "fe_model": self.fe_model.state_dict(),
                    "attention": self.attention.state_dict(),
                    "linear": self.linear.state_dict(),
                    "best_score": self.best_score,
                },
                os.path.join(
                    self.ckpt_dir,
                    "best_model.ckpt",
                )
            )

    def run_epoch(self, loader, train=True, desc="train"):
        self.fe_model.train() if train else self.fe_model.eval()
        self.attention.train() if train else self.attention.eval()
        self.linear.train() if train else self.linear.eval()
        
        self.metric_calculator.reset()
        
        batch_attention_reprs = []
        batch_labels = []
        batch_meta_data = []
        for i, batch in enumerate(tqdm(loader, desc=desc)):
            images_augs, images, labels, meta_data = batch
            images_augs = images_augs.cuda()
            images = images.cuda()
            labels = labels.cuda()
            
            # Forward
            reprs = self.fe_model(images[0, ...])
            attention_reprs = self.attention(reprs, reprs, reprs)[0].mean(dim=0)[None, ...]
            
            # Collect
            batch_attention_reprs.append(attention_reprs)
            batch_labels.append(labels[0])
            batch_meta_data.append(meta_data)
            
            if ((i + 1) % self.config.finetuner_config.core_batch_size == 0) or (i == len(loader) - 1):
                batch_attention_reprs = torch.cat(batch_attention_reprs, dim=0)
                logits = self.linear(batch_attention_reprs)
                
                labels = torch.stack(batch_labels, dim=0).cuda()
                loss = nn.CrossEntropyLoss()(logits, labels)
                
                if desc == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    learning_rates = {f"lr_group_{i}": lr for i, lr in enumerate(self.scheduler.get_last_lr())}
                    # wandb.log({"lr": self.scheduler.get_last_lr()[0]})
                    wandb.log(learning_rates)
                
                self.log_losses(loss.item(), desc)
                
                # Update metrics   
                self.metric_calculator.update(
                    batch_meta_data = batch_meta_data,
                    probs = F.softmax(logits, dim=-1).detach().cpu(),
                    labels = labels.detach().cpu(),
                )
                
                batch_attention_reprs = []
                batch_labels = []
                batch_meta_data = []
        
        # Log metrics every epoch
        self.log_metrics(desc)

    # def run_epoch(self, loader, train=True, desc="train"):
    #     self.fe_model.train() if train else self.fe_model.eval()
    #     self.attention.train() if train else self.attention.eval()
    #     self.linear.train() if train else self.linear.eval()
        
    #     self.metric_calculator.reset()
        
    #     for i, batch in enumerate(tqdm(loader, desc=desc)):                        
    #         images_augs, images, labels, meta_data = batch
    #         images_augs = images_augs.cuda()
    #         images = images.cuda()
    #         labels = labels.cuda()
            
    #         # Forward
    #         reprs = self.fe_model(images[0, ...])
    #         attention_reprs = self.attention(reprs, reprs, reprs)[0].mean(dim=0)[None, ...]
    #         logits = self.linear(attention_reprs)
            
    #         loss = nn.CrossEntropyLoss()(logits, labels)
            
    #         if desc == 'train':
    #             loss.backward()
            
    #         # Update metrics   
    #         self.metric_calculator.update(
    #             batch_meta_data = [meta_data],
    #             probs = F.softmax(logits, dim=-1).detach().cpu(),
    #             labels = labels.detach().cpu(),
    #         )
            
    #         # Step optimizer
    #         if ((i + 1) % self.config.finetuner_config.core_batch_size == 0) or (i == len(loader) - 1):
    #             self.log_losses(loss.item(), desc)
                
    #             batch_sz = self.config.finetuner_config.core_batch_size if i != len(loader) - 1 else (len(loader) % self.config.finetuner_config.core_batch_size) + 1
    #             all_parameters = chain(self.fe_model.parameters(), self.attention.parameters(), self.linear.parameters())
    #             for param in all_parameters:
    #                 if param.grad is not None:
    #                     param.grad /= batch_sz
                
    #             if desc == 'train':                
    #                 self.optimizer.step()
    #                 self.scheduler.step()
    #                 self.optimizer.zero_grad()
    #                 learning_rates = {f"lr_group_{i}": lr for i, lr in enumerate(self.scheduler.get_last_lr())}
    #                 # wandb.log({"lr": self.scheduler.get_last_lr()[0]})
    #                 wandb.log(learning_rates)

                
    #     # Log metrics every epoch
    #     self.log_metrics(desc)

    # def run_epoch(self, loader, train=True, desc="train"):
    #     self.fe_model.train() if train else self.fe_model.eval()
    #     self.attention.train() if train else self.attention.eval()
    #     self.linear.train() if train else self.linear.eval()
        
    #     self.metric_calculator.reset()
        
    #     for i, batch in enumerate(tqdm(loader, desc=desc)): 
    #         images_augs, images, labels, meta_data = batch
    #         images_augs = images_augs.cuda()
    #         images = images.cuda()
    #         labels = labels.cuda()
            
    #         batch_sz, core_len, *img_shape = images.shape
            
    #         # Forward
    #         reprs = self.fe_model(images.reshape(-1, *img_shape)).reshape(batch_sz, core_len, -1)
    #         # attention_reprs = self.attention(reprs, reprs, reprs)[0].mean(dim=1)
    #         attention_reprs = self.attention(reprs).mean(dim=1)
    #         logits = self.linear(attention_reprs)
            
    #         loss = nn.CrossEntropyLoss()(logits, labels)
            
    #         if desc == 'train':
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()
    #             self.scheduler.step()
    #             learning_rates = {f"lr_group_{i}": lr for i, lr in enumerate(self.scheduler.get_last_lr())}
    #             learning_rates.update({"epoch": self.epoch})
    #             wandb.log(learning_rates)
                
            
    #         # Update metrics   
    #         self.metric_calculator.temp_super_update(
    #             batch_meta_data = meta_data,
    #             probs = F.softmax(logits, dim=-1).detach().cpu(),
    #             labels = labels.detach().cpu(),
    #         )
                
    #     # Log metrics every epoch
    #     self.log_metrics(desc)


class TimmFeatureExtractorWrapper(nn.Module):
    def __init__(self, timm_model):
        super(TimmFeatureExtractorWrapper, self).__init__()
        self.model = timm_model
        
    def forward(self, x):
        features = self.model(x)
        return features[-1]  # Return only the last feature map
    

if __name__ == '__main__': 
    CoreFinetuneExperiment.submit()