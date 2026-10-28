import torch
import torch.nn as nn
from models.backbone.custom3dnet import Custom3DNet 
from models.backbone.r2plus1d import R2Plus1DModel
from models.backbone.resnet import ResNetModel
from models.backbone.fcnet import TabularEncoder
from models.backbone.mlpblock import MLPEncoder
from models.modulation.affinmodulation import AffineModulation
from models.modulation.gammabetaproj import GammaBetaGenerator
from models.selfattention.videoattblock import VideoAttentionBlock


class MainModel(nn.Module):
    """
    Builds video encoder + tabular encoder + fusion + heads from YAML.
    """
    def __init__(self, cfg, task_info: dict):
        super().__init__()
        mcfg = cfg["model"]
        self.task_info = task_info
        self.model_name = mcfg.get("name", "multimodal")
        self.use_tabular = mcfg.get("use_tabular", True)
        self.use_video = mcfg.get("use_video", True)

        if self.model_name not in ["videoonly", "multimodal"]:
            raise ValueError(f"❌ Unknown model name: {self.model_name}")

        # ---------------------- VIDEO BACKBONE --------------------
        backbone_name = mcfg.get("video_backbone", "custom3dnet").lower()
        # Add other backbones here if you want (R2Plus1D, ResNet3D, etc)
        if backbone_name == "custom3dnet":
            self.video_encoder = Custom3DNet(mcfg)
        else:
            raise ValueError(f"❌ Unknown video backbone: {backbone_name}")
        
        video_channels = self.video_encoder.out_channels

        # ---------------------- TABULAR BACKBONE -----------------
        if self.model_name == "multimodal":
            self.tabular_encoder = TabularEncoder(mcfg["tabular"])
            Dt = self.tabular_encoder.out_dim
            self.gamma_beta_proj = GammaBetaGenerator(Dt, video_channels)
            self.modulation = AffineModulation(mcfg["affine_modulation"])
        else:
            self.tabular_encoder = None
            self.gamma_beta_proj = None
            self.modulation = None

        # -------------------- VIDEO ATTENTION BLOCK ---------------
        self.video_att = VideoAttentionBlock(
            in_channels=self.video_encoder.out_dim, cfg=mcfg)
        
        # --------------------------- HEADS ------------------------
        pred_cfg = mcfg.get("predhead", {})
        self.multitask_head = bool(pred_cfg.get("multitask_head", True))
        self.heads = nn.ModuleDict({
            task: MLPEncoder({       
                "input_dim": pred_cfg.get("input_dim", 64),
                "hidden_dims": pred_cfg.get("hidden_dims", [128]),
                "out_dim": pred_cfg.get("out_dim", 1),
                "activation": pred_cfg.get("activation", "relu"),
                "normalization": pred_cfg.get("normalization", "batchnorm"),
                "dropout": pred_cfg.get("dropout", 0.3),
            })
            for task in task_info
        })

    # ---------------------- FORWARD ----------------------
    def forward(self, video=None, tabular=None):

        # 1. Video Only
        if self.model_name == "videoonly":
            
            if video is None:
                raise ValueError("video is required for videoonly mode")
            
            feat = self.video_encoder(video)
            v = self.video_att(feat)
        
        # 3. Multimodal Fusion
        elif self.model_name == "multimodal":
            
            if video is None:
                raise ValueError("video is required for multimodal mode")
            if tabular is None:
                raise ValueError("tabular is required for multimodal mode")
            
            feat = self.video_encoder(video)         # (B, Dv, T, H, W)
            t = self.tabular_encoder(tabular)        # (B, Dt)
            gamma, beta = self.gamma_beta_proj(t)    # (B, Dv), (B, Dv)
            a = self.modulation(feat, gamma, beta)   # (B, Dv, T, H, W)
            v = self.video_att(a)                    # (B, Dv)
            

        # 4. Heads
        if self.multitask_head:
            return {task: head(v) for task, head in self.heads.items()}

        (only_task, only_head), = self.heads.items()
        return only_head(v)