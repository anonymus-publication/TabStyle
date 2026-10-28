import os, cv2, torch, numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, Tuple
from tqdm import tqdm

from utils.transform import compute_video_mean_std, load_stats, save_stats
from utils.load_video import load_video  
from utils.sample_pad import sample_clip_and_pad 
from utils.augment import RandomSpatialShift, RandomSpatialRotation

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
KINETICS_STD  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)


class EchoDataset(Dataset):
    def __init__(self, video_dir: str, tabular_df: pd.DataFrame, task_type: Dict[str, str],
                 
                 target_normalizer=None, target_regression_cols: Tuple[str] = (), target_binary_cols: Tuple[str] = (),
                 tabular_normalizer=None, tabular_regression_cols: Tuple[str] = (), tabular_binary_cols: Tuple[str] = (),
                 tabular_cols=None,
                 
                 size: Tuple[int, int] = (112, 112),
                 clip_len: int = 32, sampling: str = "center", frame_stride: int = 1,
                 normalization: str = "imagenet", dataset_stats: Tuple[torch.Tensor, torch.Tensor] = None,
                 stats_cache_path: str = None, compute_stats_if_missing: bool = False,
                 stats_sample_size: int = 200, grayscale: bool = False,
                 extensions: Tuple[str] = ("avi",), show_tqdm: bool = False, use_video=True, use_tabular=False,
                 is_train: bool = True, aug_shift: int = 8):

        self.use_video = use_video
        self.use_tabular = use_tabular
        self.is_train = is_train
        # self.spatial_aug = RandomSpatialShift(max_shift=8)
        self.spatial_aug = RandomSpatialShift(max_shift=aug_shift)
        self.spatial_rot = RandomSpatialRotation(max_degree=15.0)

        # ----------------------------------------------------------
        # FIX 1 — define tabular_df FIRST
        # ----------------------------------------------------------
        self.tabular_df = tabular_df.set_index("video_id")

        # ----------------------------------------------------------
        # FIX 2 — now build video paths
        # ----------------------------------------------------------
        if self.use_video:
            # real list of video files
            self.video_paths = sorted([
                os.path.join(video_dir, f) for f in os.listdir(video_dir)
                if any(f.lower().endswith(ext) for ext in extensions)
            ])
            # keep only videos that exist in tabular
            self.video_paths = [
                vp for vp in self.video_paths
                if os.path.splitext(os.path.basename(vp))[0] in self.tabular_df.index
            ]
        else:
            # No video — just index by tabular rows
            self.video_paths = [
                os.path.join(video_dir, f"{vid}.avi")  
                for vid in self.tabular_df.index
            ]
        # ----------------------------------------------------------

        self.tabular_cols = tabular_cols
        self.task_type = task_type
        # New
        self.target_normalizer = target_normalizer
        self.target_regression_cols = list(target_regression_cols)
        self.target_binary_cols = list(target_binary_cols)

        self.tabular_normalizer = tabular_normalizer
        self.tabular_regression_cols = list(tabular_regression_cols)
        self.tabular_binary_cols = list(tabular_binary_cols)
        self.tabular_cols = tabular_cols

        self.size, self.clip_len = size, clip_len
        self.sampling, self.frame_stride = sampling, frame_stride
        self.grayscale, self.show_tqdm = grayscale, show_tqdm
        self.normalization = normalization

        # ------------------ Video normalization ------------------
        if normalization == "imagenet":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        elif normalization == "kinetics":
            mean, std = KINETICS_MEAN, KINETICS_STD
        elif normalization == "dataset":
            if dataset_stats:
                mean, std = dataset_stats
            elif stats_cache_path and os.path.isfile(stats_cache_path):
                mean, std = load_stats(stats_cache_path)
            elif compute_stats_if_missing:
                if self.is_train:
                    mean, std = compute_video_mean_std(
                        self.video_paths,
                        sample_size=stats_sample_size,
                        size=size,
                        grayscale=grayscale,
                        frame_stride=frame_stride,
                        progress=True,
                    )
                    if stats_cache_path:
                        save_stats(mean, std, stats_cache_path)
                else:
                    raise ValueError("Cannot compute dataset stats on val/test set. Please provide stats_cache_path with precomputed stats or set compute_stats_if_missing=False.")
            else:
                raise ValueError("Missing dataset stats and cannot compute.")
        else:
            mean, std = None, None

        C = 1 if grayscale else 3
        if mean is not None:
            mean = list(mean)
            std = list(std)

            if len(mean) == 1 and C == 3:
                mean, std = mean * 3, std * 3
            elif len(mean) == 3 and C == 1:
                mean, std = [float(np.mean(mean))], [float(np.mean(std))]
            elif len(mean) != C:
                raise ValueError(f"Normalization stats have {len(mean)} channels but input expects {C}.")

            self.mean = torch.tensor(mean, dtype=torch.float32).view(C, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(C, 1, 1)
        else:
            self.mean, self.std = None, None


    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        row = self.tabular_df.loc[video_id]
        video_name = row["video_name"]

        # --------------- VIDEO ----------------
        if self.use_video:
            frames, fps = load_video(path=video_path, video_dim=self.size,
                                     grayscale=self.grayscale,
                                     show_tqdm=self.show_tqdm)

            if self.frame_stride > 1:
                frames = frames[::self.frame_stride]

            frames, len_mask = sample_clip_and_pad(frames,
                                                   self.clip_len,
                                                   method=self.sampling)

            if self.is_train:
                frames = self.spatial_aug(frames)

            if self.mean is not None:
                frames = (frames - self.mean) / self.std

            frames_out = frames.permute(1, 0, 2, 3)

        else:
            frames_out = None
            fps = None
            len_mask = None

        # ---------------- TABULAR ----------------
        # New
        if self.tabular_cols is not None and len(self.tabular_cols) > 0:
            feature_map = {}

            if len(self.tabular_regression_cols) > 0:
                reg_df = pd.DataFrame(
                    [row[self.tabular_regression_cols].values],
                    columns=self.tabular_regression_cols
                )
                if self.tabular_normalizer is not None:
                    reg_vals = self.tabular_normalizer.transform(reg_df).squeeze(0)
                else:
                    reg_vals = reg_df.values.squeeze(0)

                for i, col in enumerate(self.tabular_regression_cols):
                    feature_map[col] = float(reg_vals[i])

            for col in self.tabular_binary_cols:
                feature_map[col] = float(row[col])

            tabular = np.array([feature_map[col] for col in self.tabular_cols], dtype=np.float32)
        else:
            tabular = np.empty(0, dtype=np.float32)

        tabular = torch.as_tensor(tabular, dtype=torch.float32)
        # ---------------- TARGETS ----------------
        targets, label_masks = {}, {}
        # New
        # precompute normalized regression targets once
        norm_target_map = {}
        if len(self.target_regression_cols) > 0:
            target_reg_df = pd.DataFrame(
                [row[self.target_regression_cols].values],
                columns=self.target_regression_cols
            )
            if self.target_normalizer is not None:
                norm_target_vals = self.target_normalizer.transform(target_reg_df).squeeze(0)
            else:
                norm_target_vals = target_reg_df.values.squeeze(0)

            norm_target_map = {
                col: float(norm_target_vals[i])
                for i, col in enumerate(self.target_regression_cols)
            }

        for task, ttype in self.task_type.items():
            val = row.get(task, None)

            if pd.isna(val):
                label_masks[task] = torch.tensor(0.0, dtype=torch.float32)
                targets[task] = torch.tensor(0.0, dtype=torch.float32)
            else:
                label_masks[task] = torch.tensor(1.0, dtype=torch.float32)

                if ttype == "regression":
                    targets[task] = torch.tensor(norm_target_map[task], dtype=torch.float32)
                else:
                    targets[task] = torch.tensor(float(val), dtype=torch.float32)

        return {
            "frames": frames_out,
            "len_mask": len_mask,
            "tabular": tabular,
            "targets": targets,
            "label_masks": label_masks,
            "video_id": video_id,
            "video_name": video_name,
            "fps": fps,
            "num_frames": frames_out.shape[1] if frames_out is not None else 0,
        }
