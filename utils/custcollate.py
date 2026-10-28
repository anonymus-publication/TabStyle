import torch
from torch.utils.data._utils.collate import default_collate

def multitask_collate(batch):
    collated = {}

    # -------------------------------
    # VIDEO FRAMES (safe handling)
    # -------------------------------
    frames_list = [b["frames"] for b in batch]

    if frames_list[0] is None:
        # If video disabled → keep None
        collated["frames"] = None
    else:
        # Normal collate
        collated["frames"] = default_collate(frames_list)
    
    

    # -------------------------------
    # TABULAR
    # -------------------------------
    collated["tabular"] = default_collate([b["tabular"] for b in batch])

    # -------------------------------
    # TARGETS
    # -------------------------------
    collated["targets"] = {
        key: default_collate([b["targets"][key] for b in batch])
        for key in batch[0]["targets"]
    }

    # -------------------------------
    # LABEL MASKS
    # -------------------------------
    collated["label_masks"] = {
        key: default_collate([b["label_masks"][key] for b in batch])
        for key in batch[0]["label_masks"]
    }

    # -------------------------------
    # METADATA
    # -------------------------------
    collated["video_id"] = [b["video_id"] for b in batch]
    collated["video_name"] = [b["video_name"] for b in batch]
    collated["fps"] = [b["fps"] for b in batch]
    collated["num_frames"] = [
        (0 if b["num_frames"] is None else b["num_frames"]) 
        for b in batch
    ]

    return collated
