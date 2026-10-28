import os
import json
from pyexpat import model
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold, KFold
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb as wandb
from tqdm import tqdm
import copy
from joblib import dump
import time 
import cv2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# local imports
from utils.task_type import infer_task_types
from data.loader import EchoDataset
from utils.losses import compute_multitask_loss
from models.backbone.r2plus1d import R2Plus1DModel
from models.backbone.resnet import ResNetModel
from models.backbone.custom3dnet import Custom3DNet
from models.mainmodel import MainModel
from utils.transform import compute_video_mean_std
from utils.custcollate import multitask_collate
from utils.sanitycheck import sanity_check
from utils.save_split import save_split_csv
# from saliency.saliency import input_gradient_saliency, integrated_gradients_video
# from saliency.debug_sal import saliency_to_heatmap, overlay_heatmap
from data.preprocess import balance_videos_globally
from training.dataloader import build_loaders_from_cfg
from training.trainmain import run_train
from training.evalmain import run_eval


# ===============================================================
# ============================ Utils ============================
# ===============================================================

# ---------------------------- SEED -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------- Model factory ----------------
def create_model_from_cfg(model_cfg: dict, task_info: dict) -> nn.Module:
    name = str(model_cfg.get("name", "")).lower()
    if name in ("r2plus1d", "r(2+1)d", "r2p1d"):
        return R2Plus1DModel(**model_cfg)
    elif name in ("resnet3d", "resnet"):
        return ResNetModel(**model_cfg)
    elif name in ("custom3d", "custom3dnet"):
        return Custom3DNet(**model_cfg)
    elif name in ("multimodal", "multimodel"):
        return MainModel(cfg={"model": model_cfg}, task_info=task_info)
    elif name in ("videoonly", "videonly"):
        return MainModel(cfg={"model": model_cfg}, task_info=task_info)
    else:
        raise ValueError(f"Unknown model type '{name}' in model_cfg")

# ---------------- Main Function ----------------
def main_from_config(cfg_or_path: str):
    # New
    # with open(cfg_path, "r") as f:
    #     cfg = yaml.safe_load(f)
    if isinstance(cfg_or_path, str):
        with open(cfg_or_path, "r") as f:
            cfg = yaml.safe_load(f)
    elif isinstance(cfg_or_path, dict):
        cfg = copy.deepcopy(cfg_or_path)
    else:
        raise TypeError("cfg_or_path must be a path or a dict")

    seed_everything(cfg.get("seed", 42))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cfg.get("cpu", False) else "cpu"
    )

    # New
    # if cfg.get("wandb", {}).get("enable", False):
    if cfg.get("wandb", {}).get("enable", False) and wandb.run is None:
        wandb.init(
            project=cfg["wandb"].get("project", "echo-project"),
            name=cfg["wandb"].get("name", None),
            config=cfg,
        )

    # build Dataset
    loaders, task_info, target_regression_cols, target_normalizer, tabular_normalizer = build_loaders_from_cfg(cfg)
    
    # Save splits based on balancing mode and seed
    balance_mode = cfg.get("balance", {}).get("mode", "none")
    seed = cfg.get("seed", 42)

    split_csv_path = Path("splits") / f"patient_splits_{balance_mode}_seed{seed}.csv"

    save_split_csv(loaders, split_csv_path)

    # Build Model
    model = create_model_from_cfg(cfg["model"], task_info).to(device)

    # Sanity Check
    if cfg.get("sanity_check", False):
        sanity_check(model, loaders, device)

    # Get Training Options
    loss_weights = cfg["train"].get("loss_weights", None)
    phase = str(cfg.get("phase", "train")).lower()
    use_tqdm = cfg["train"].get("show_tqdm", False)

    best_ckpt = None
    best_info = None

    if phase == "train":
        print("=============== Starting Training ===============")
        if loaders["train"] is None:
            raise RuntimeError("No TRAIN split available.")

        best_ckpt, best_info = run_train(cfg, model, loaders, task_info, loss_weights, device, use_tqdm)
        # print("DEBUG best_info:")
        # print(json.dumps(best_info, indent=2, default=str))

        # print("DEBUG flattened best_info:")
        # print(flatten_best_info_for_csv(best_info))

        print("=============== Starting Testing ===============")
        if cfg["train"].get("eval_after_train", True) and loaders["test"] is not None:
            cfg.setdefault("eval", {})["checkpoint"] = str(best_ckpt)
            run_eval(
                cfg, model, loaders["test"], task_info, loss_weights, device,
                split="test",
                inverse_transform=True,
                reg_pipeline=target_normalizer,
                regression_cols=target_regression_cols
            )
        
        return {
            "best_checkpoint": str(best_ckpt),
            "best_info": best_info,
        }

    elif phase == "val":
        print("=============== Starting Validation ===============")
        if loaders["val"] is None:
            raise RuntimeError("No VAL split available.")

        run_eval(
            cfg, model, loaders["val"], task_info, loss_weights, device,
            split="val",
            inverse_transform=False,
            reg_pipeline=target_normalizer,
            regression_cols=target_regression_cols
        )

        return {
            "best_checkpoint": None,
            "best_info": None,
        }

    elif phase == "test":
        print("=============== Starting Testing ===============")
        if loaders["test"] is None:
            raise RuntimeError("No TEST split available.")

        run_eval(
            cfg, model, loaders["test"], task_info, loss_weights, device,
            split="test",
            inverse_transform=True,
            reg_pipeline=target_normalizer,
            regression_cols=target_regression_cols
        )
        return {
            "best_checkpoint": None,
            "best_info": None,
        }

    elif phase == "saliency":
        print("=============== Starting Saliency Generation ===============")

        for split in ["train", "val", "test"]:
            if loaders.get(split) is None:
                print(f"[Saliency] Skipping {split}: loader is None")
                continue

            run_eval(
                cfg,
                model,
                loaders[split],
                task_info,
                loss_weights,
                device,
                split=split,
                inverse_transform=False,
                reg_pipeline=target_normalizer,
                regression_cols=target_regression_cols,
            )

        return {
            "best_checkpoint": cfg.get("eval", {}).get("checkpoint", None),
            "best_info": None,
        }
        
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Config-driven Echo Trainer")
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    main_from_config(args.config)