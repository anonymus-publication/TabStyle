import torch
import torch.nn as nn
import torchvision.models as models

class ResNetModel(nn.Module):
    def __init__(self, task_info, modelname="resnet18", pretrained=True, freeze_backbone=False):
        super().__init__()
        self.task_info = task_info

        # Load 2D CNN backbone
        if modelname not in models.__dict__:
            raise ValueError(f"Model '{modelname}' not found in torchvision.models")

        self.backbone = models.__dict__[modelname](pretrained=pretrained)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove final classification layer

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Multi-task heads
        self.heads = nn.ModuleDict()
        for task, ttype in task_info.items():
            self.heads[task] = nn.Linear(self.feature_dim, 1)

    def forward(self, x):
        """
        x: Tensor of shape (B, C, H, W) or (B*T, C, H, W)
        If using frame-level inference, you can mean-pool outside the model.
        """
        features = self.backbone(x)  # shape: (B, F)
        outputs = {}

        for task, head in self.heads.items():
            outputs[task] = head(features).squeeze(1)  # shape: (B,)

        return outputs
