import torch
import os
import torch.nn as nn
import torchvision.models.video as models

# NOTE:
# r2plus1d_18, r3d_18, and mc3_18 are 3D video models, pretrained only on Kinetics-400

class R2Plus1DModel(nn.Module):
    def __init__(self, task_info, modelname="r2plus1d_18", pretrained=True, freeze_backbone=False):
        super().__init__()
        self.task_info = task_info

        # Load 2D CNN backbone
        if modelname not in models.__dict__:
            raise ValueError(f"Model '{modelname}' not found in torchvision.models")
         
        # Backbone: spatiotemporal pretrained CNN
        self.backbone = models.__dict__[modelname](pretrained=pretrained)
        self.feature_dim = self.backbone.fc.in_features
        # model outputs the raw feature vectors
        self.backbone.fc = nn.Identity()
        self.attention = SelfAttention(dim=self.feature_dim, heads=4, dropout=0.1)

        if pretrained:
            os.makedirs("./outputs", exist_ok=True)
            torch.save(self.backbone.state_dict(), os.path.join("output", f"{modelname}_pretrained_weights.pth"))

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Multi-head setup
        # Each head is just a Linear(512, 1) regardless of whether it's regression or binary classification.
        # Use appropriate loss functions later based on task_info.
        self.heads = nn.ModuleDict()
        for task, ttype in task_info.items():
            self.heads[task] = nn.Linear(self.feature_dim, 1)

    def forward(self, x):
        features = self.backbone(x)  # shape: (B, F)
        features = self.attention(features)  # shape: (B, 512)
        outputs = {}

        for task, head in self.heads.items():
            outputs[task] = head(features).squeeze(1)  # shape: (B,)

        return outputs
