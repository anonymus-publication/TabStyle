from pathlib import Path
import os
import pandas as pd
import random
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
from joblib import dump

# Internal imports
from data.loader import EchoDataset
from utils.transform import compute_video_mean_std
from data.preprocess import balance_videos_globally
from utils.task_type import infer_task_types
from utils.custcollate import multitask_collate

# ===============================================================
# ============================ Utils ============================
# ===============================================================

def print_video_level_stats(df_split, split_name, label_col):
    labels = df_split.groupby("video_name")[label_col].first()
    counts = labels.value_counts().sort_index()
    ratios = counts / counts.sum()

    print(f"\n{split_name.upper()} VIDEO-LEVEL CLASS DISTRIBUTION:")
    for cls in counts.index:
        print(f"  Class {cls}: {counts[cls]} videos ({ratios[cls]*100:.1f}%)")

def print_clip_level_stats(df_split, split_name, label_col):
    counts = df_split[label_col].value_counts().sort_index()
    ratios = counts / counts.sum()

    print(f"\n{split_name.upper()} CLIP-LEVEL DISTRIBUTION:")
    for cls in counts.index:
        print(f"  Class {cls}: {counts[cls]} clips ({ratios[cls]*100:.1f}%)")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ===============================================================
# ==================== Main Function ============================
# ===============================================================

def build_loaders_from_cfg(cfg):
    print("==> Loading Tabular Dataset:", cfg["data"]["csv_path"])
    df = pd.read_csv(cfg["data"]["csv_path"])

    print("==> Loading Video Dataset:", cfg["data"]["video_root"])
    video_root = Path(cfg["data"]["video_root"])

    print("\n===== Pair Checking Video and Tabular =========")

    # Normalize video IDs from disk
    existing_video_ids = {
        p.stem.lower().strip()
        for p in video_root.glob("*.avi")
    }

    # Normalize video_id column in CSV
    df["video_id_norm"] = (
        df["video_id"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(".avi", "", regex=False)
    )

    # Keep only paired rows
    df_paired = df[df["video_id_norm"].isin(existing_video_ids)].copy()

    print(f"Original rows: {len(df)}")
    print(f"Paired rows:   {len(df_paired)}")
    print(f"Dropped rows:  {len(df) - len(df_paired)}")

    # Optional but VERY useful diagnostics
    print(f"Original unique videos: {df['video_id_norm'].nunique()}")
    print(f"Paired unique videos:   {df_paired['video_id_norm'].nunique()}")

    print("=============== Building Dataset ================")
    # df = pd.read_csv(cfg["data"]["csv_path"])
    df = df_paired.copy()
    task_cols = cfg["data"]["task_cols"]
    tabular_cols = cfg["data"].get("tabular_features", None)

    # ------------ TASK TYPE INFERENCE (on full df) -------------
    # New
    task_type = infer_task_types(df, task_cols, exclude_cols=["video_id"])
    target_regression_cols = [t for t in task_type if task_type[t] == "regression"]
    target_binary_cols     = [t for t in task_type if task_type[t] == "binary"]

    if tabular_cols is not None and len(tabular_cols) > 0:
        tabular_type = infer_task_types(df, tabular_cols, exclude_cols=["video_id"])
        tabular_regression_cols = [c for c in tabular_type if tabular_type[c] == "regression"]
        tabular_binary_cols     = [c for c in tabular_type if tabular_type[c] == "binary"]
    else:
        tabular_type = {}
        tabular_regression_cols = []
        tabular_binary_cols = []

    # -------- GLOBAL VIDEO-LEVEL BALANCING BEFORE SPLIT---------
    if cfg.get("balance", {}).get("mode") == "video_global":
        target_col = cfg["balance"]["target_col"]
        df = balance_videos_globally(df, target_col, seed=cfg.get("seed", 42))

    # ----------- SPLIT BY UNIQUE video_name, NOT ROWS------------
    unique_videos = df["video_name"].unique()

    val_ratio  = float(cfg["data"].get("val_ratio", 0.15))
    test_ratio = float(cfg["data"].get("test_ratio", 0.15))
    
    # Seeding 
    seed = cfg.get("seed", 42)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1)
    test_generator = torch.Generator()
    test_generator.manual_seed(seed + 2)    

    # stratify by label if requested
    if "stratify_by" in cfg["data"]:
        strat_col = df.groupby("video_name")[cfg["data"]["stratify_by"]].first()
        train_videos, rest_videos = train_test_split(
            unique_videos, 
            test_size=val_ratio + test_ratio,
            random_state=seed,
            stratify=strat_col.loc[unique_videos]
        )
        rest_labels = strat_col.loc[rest_videos]
        val_size = val_ratio / (val_ratio + test_ratio)
        val_videos, test_videos = train_test_split(
            rest_videos,
            test_size=1 - val_size,
            random_state=seed,
            stratify=rest_labels
        )
    else:
        train_videos, rest_videos = train_test_split(
            unique_videos,
            test_size=val_ratio + test_ratio,
            random_state=seed
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        val_videos, test_videos = train_test_split(
            rest_videos,
            test_size=1 - val_size,
            random_state=seed
        )

    # attach clips back
    train_df = df[df["video_name"].isin(train_videos)].reset_index(drop=True)
    val_df   = df[df["video_name"].isin(val_videos)].reset_index(drop=True)
    test_df  = df[df["video_name"].isin(test_videos)].reset_index(drop=True)

    print("==> Split Sizes (video-level):")
    print(f" Train videos: {len(train_videos)} | Clips: {len(train_df)}")
    print(f" Val videos:   {len(val_videos)}   | Clips: {len(val_df)}")
    print(f" Test videos:  {len(test_videos)}  | Clips: {len(test_df)}")

    # -------------LEAKAGE CHECK-----------------------------
    train_vids = set(train_df["video_name"])
    val_vids   = set(val_df["video_name"])
    test_vids  = set(test_df["video_name"])

    assert train_vids.isdisjoint(val_vids), \
        f"❌ DATA LEAKAGE DETECTED: {train_vids.intersection(val_vids)} appear in BOTH train and val sets!"

    assert train_vids.isdisjoint(test_vids), \
        f"❌ DATA LEAKAGE DETECTED: {train_vids.intersection(test_vids)} appear in BOTH train and test sets!"

    assert val_vids.isdisjoint(test_vids), \
        f"❌ DATA LEAKAGE DETECTED: {val_vids.intersection(test_vids)} appear in BOTH val and test sets!"

    print('✓ No data leakage detected between splits (video-level separation OK).')

    # ------------TABULAR NORMALIZER ONLY ON TRAIN----------------
    # New 
    # TARGET NORMALIZER (only continuous target cols)
    if len(target_regression_cols) > 0:
        target_normalizer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            # ("scaler", StandardScaler())
            ("scaler", MinMaxScaler())
        ])
        target_normalizer.fit(train_df[target_regression_cols])
        dump(target_normalizer, "target_normalizer.joblib")
    else:
        target_normalizer = None

    # TABULAR FEATURE NORMALIZER (only continuous tabular cols) 
    if tabular_cols is not None and len(tabular_regression_cols) > 0:
        tabular_normalizer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            # ("scaler", StandardScaler())
            ("scaler", MinMaxScaler())
        ])
        tabular_normalizer.fit(train_df[tabular_regression_cols])
        dump(tabular_normalizer, "tabular_feature_normalizer.joblib")
    else:
        tabular_normalizer = None

    # -------------VIDEO NORMALIZATION-----------------------------
    normalization = cfg["data"].get("normalization", "imagenet")
    if normalization == "dataset":
        video_paths = [
            os.path.join(cfg["data"]["video_root"], f"{vid}.avi")
            for vid in train_df["video_id"].values
        ]
        mean, std = compute_video_mean_std(video_paths)
        dataset_stats = (mean, std)
    else:
        dataset_stats = None

    # -----------DATASETS AND LOADERS-------------------------------
    use_video = cfg["model"].get("use_video", True)
    use_tabular = cfg["model"].get("use_tabular", True)
    num_workers = cfg["train"]["num_workers"]
    batch_size = cfg["train"]["batch_size"]

    def make_ds(df_split, is_train: bool):
        return EchoDataset(
            video_dir=cfg["data"]["video_root"],
            tabular_df=df_split,
            task_type=task_type,
            
            # New
            target_normalizer=target_normalizer,
            target_regression_cols=target_regression_cols,
            target_binary_cols=target_binary_cols,

            tabular_normalizer=tabular_normalizer,
            tabular_regression_cols=tabular_regression_cols,
            tabular_binary_cols=tabular_binary_cols,
            tabular_cols=tabular_cols,
            
            size=cfg["data"].get("size", (112,112)),
            clip_len=cfg["data"].get("clip_len", 32),
            sampling=cfg["data"].get("sampling", "center"),
            normalization=normalization,
            dataset_stats=dataset_stats,
            frame_stride=cfg["data"].get("frame_stride", 1),
            grayscale=cfg["data"].get("grayscale", False),
            use_video=use_video,
            use_tabular=use_tabular,
            is_train=is_train,
            aug_shift=cfg["data"].get("aug_shift", 8)

        )

    loaders = {
        "train": DataLoader(make_ds(train_df, is_train=True), batch_size=batch_size,
                            shuffle=True,  num_workers=num_workers, pin_memory=True,
                            collate_fn=multitask_collate, worker_init_fn=seed_worker, 
                            generator=train_generator, persistent_workers=(num_workers > 0)),
        "val":   DataLoader(make_ds(val_df, is_train=False),   batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True,
                            collate_fn=multitask_collate, worker_init_fn=seed_worker, 
                            generator=val_generator, persistent_workers=(num_workers > 0)),
        "test":  DataLoader(make_ds(test_df, is_train=False),  batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True,
                            collate_fn=multitask_collate, worker_init_fn=seed_worker, 
                            generator=test_generator, persistent_workers=(num_workers > 0))
    }

    label_col = cfg["data"]["task_cols"][1]

    print_video_level_stats(train_df, "train", label_col)
    print_video_level_stats(val_df,   "val",   label_col)
    print_video_level_stats(test_df,  "test",  label_col)

    print_clip_level_stats(train_df, "train", label_col)
    print_clip_level_stats(val_df,   "val",   label_col)
    print_clip_level_stats(test_df,  "test",  label_col)

    return loaders, task_type, target_regression_cols, target_normalizer, tabular_normalizer