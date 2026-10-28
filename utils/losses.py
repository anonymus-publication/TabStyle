import torch.nn as nn
import torch

def compute_multitask_loss(outputs, targets, task_info, loss_weights=None, agg: str = "mean"):
    """
    Multi-task loss with configurable aggregation.

    Args:
        outputs: dict of {task: prediction tensor}
        targets: dict of {task: target tensor}
        task_info: dict of {task: "binary" | "regression"}
        loss_weights: dict of {task: weight} or None (default = all 1.0)
        agg: "mean" (default) or "sum"
    """
    losses = {}

    if loss_weights is None:
        loss_weights = {task: 1.0 for task in task_info}

    task_losses = []
    for task, ttype in task_info.items():
        pred = outputs[task].view(-1)
        true = targets[task].view(-1).float()

        if ttype == "binary":
            loss = nn.BCEWithLogitsLoss()(pred, true)
        elif ttype == "regression":
            # loss = nn.MSELoss()(pred, true)
            loss = nn.L1Loss()(pred, true)
        else:
            raise ValueError(f"Unknown task type: {ttype}")

        weighted = loss_weights.get(task, 1.0) * loss
        losses[task] = weighted.item()
        task_losses.append(weighted)

    if not task_losses:
        total_loss = torch.tensor(0.0)
    else:
        if agg == "mean":
            total_loss = torch.stack(task_losses).mean()
        elif agg == "sum":
            total_loss = torch.stack(task_losses).sum()
        else:
            raise ValueError(f"Unknown loss aggregation: {agg}")

    return total_loss, losses
