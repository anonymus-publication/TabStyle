# utils/augment.py
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random

class RandomSpatialShift:
    """
    Randomly shifts a video clip spatially by up to max_shift pixels.
    Applies the SAME shift to all frames (critical for medical videos).
    
    Input shape: (T, C, H, W)
    Output shape: (T, C, H, W)
    """
    def __init__(self, max_shift: int = 8):
        self.max_shift = max_shift

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # video: (T, C, H, W)
        dx = random.randint(-self.max_shift, self.max_shift)
        dy = random.randint(-self.max_shift, self.max_shift)

        shifted = []
        for frame in video:
            frame = TF.affine(
                frame,
                angle=0.0,
                translate=(dx, dy),
                scale=1.0,
                shear=(0.0, 0.0)
            )
            shifted.append(frame)

        return torch.stack(shifted, dim=0)




class RandomSpatialRotation:
    """
    Randomly rotates a video clip by up to max_degree degrees.
    Applies the SAME rotation to all frames (critical for medical videos).

    Input shape: (T, C, H, W)
    Output shape: (T, C, H, W)
    """
    def __init__(self, max_degree: float = 25.0):
        self.max_degree = max_degree

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # video: (T, C, H, W)
        angle = random.uniform(-self.max_degree, self.max_degree)

        rotated = []
        for frame in video:
            frame = TF.rotate(
                frame,
                angle=angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                expand=False,
            )
            rotated.append(frame)

        return torch.stack(rotated, dim=0)
    

# augment = torchvision.transforms.Compose([
#     RandomSpatialShift(max_shift=8),
#     RandomSpatialRotation(max_degree=25),
# ])


