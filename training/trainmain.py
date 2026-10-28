import time
import numpy as np
import torch
import wandb
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from tqdm import tqdm
import copy
from utils.losses import compute_multitask_loss
from training.evalmain import evaluate_model

# ===============================================================
# =================== Utils ================================
# ===============================================================

# ---------------------- To Device Batch ----------------------
def to_device_batch(batch: dict, device: torch.device, use_video: bool = True, use_tabular: bool = True):
    inputs = {}
    # VIDEO 
    if use_video:
        # Always present in batch["frames"], but only moved if needed
        inputs["video"] = batch["frames"].to(device, non_blocking=True)
    else:
        inputs["video"] = None  # Explicitly mark inactive branch
    # TABULAR 
    if use_tabular:
        tab = batch.get("tabular", None)
        if tab is not None:
            tab = tab.to(device, non_blocking=True)
        inputs["tabular"] = tab
    else:
        inputs["tabular"] = None
    # TARGETS 
    targets = {k: v.to(device) for k, v in batch["targets"].items()}
    # META 
    meta = {
        "video_id": batch.get("video_id", None),
        "video_name": batch.get("video_name", None),
        "fps": batch.get("fps", None),
        "num_frames": batch.get("num_frames", None),
    }
    return inputs, targets, meta

# ------------------ Get Nested Value ------------------
def get_nested_value(d, dotted_key, default=None):
    keys = dotted_key.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# ===============================================================
# =================== Train main ================================
# ===============================================================

# ---------------- Training Epoche ----------------
def train_one_epoch(model, loader, optimizer, device, loss_fn, task_info, loss_weights, use_tqdm=False, 
                    epoch=None, agg="mean", threshold=0.5):
    model.train()
    running_loss = 0.0
    all_targets, all_preds, all_probs = {}, {}, {}
    iterator = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False) if use_tqdm else loader

    for batch_idx, batch in enumerate(iterator):
        inputs, targets, _ = to_device_batch(batch, device, model.use_video, model.use_tabular)
        optimizer.zero_grad()
        if model.use_video and model.use_tabular:
            outputs = model(inputs["video"], inputs["tabular"])
        elif model.use_video:
            outputs = model(inputs["video"], None)
        elif model.use_tabular:
            outputs = model(None, inputs["tabular"])
        else:
            raise RuntimeError("Both video and tabular disabled — cannot train.")
        loss, task_losses = loss_fn(outputs, targets, task_info, loss_weights, agg=agg)
        
        # if batch_idx % 50 == 0:
        #     print(f"\n[Epoch {epoch} | Batch {batch_idx}] task losses:")
        #     for t, v in task_losses.items():
        #         print(f"   {t}: {v:.4f}")

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        for task, y_true in targets.items():
            y_true_np = y_true.detach().cpu().numpy().ravel().tolist()
            all_targets.setdefault(task, []).extend(y_true_np)
            y_out = outputs[task]
            if task_info[task] == 'binary':
                if y_out.shape[-1] == 1:
                    prob = torch.sigmoid(y_out).detach().cpu().numpy().ravel()

                elif y_out.shape[-1] == 2:
                    logit = (y_out[..., 1] - y_out[..., 0])
                    prob = torch.sigmoid(logit).detach().cpu().numpy().ravel()
                else:
                    prob = torch.sigmoid(y_out).detach().cpu().numpy().ravel()

                pred = (prob >= threshold).astype(float)
                all_probs.setdefault(task, []).extend(prob.tolist())
                all_preds.setdefault(task, []).extend(pred.tolist())
            else:
                pred = y_out.detach().cpu().numpy().ravel()
                all_preds.setdefault(task, []).extend(pred.tolist())

    # ---- average loss ----
    avg_loss = running_loss / max(1, len(loader))

    # ---- Compute metrics (per-task + overall) ----
    acc_vals, mae_vals = [], []
    train_stats = {}

    for task, y_true in all_targets.items():
        y_true = np.array(y_true)
        pred = np.array(all_preds[task])

        if task_info[task] == "binary":
            prob = np.array(all_probs[task])
            acc = accuracy_score(y_true, pred)
            acc_vals.append(acc)
            train_stats[f"{task}/accuracy"] = acc
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, prob)
                train_stats[f"{task}/auroc"] = auroc

        else:  # regression
            mae = mean_absolute_error(y_true, pred)
            mae_vals.append(mae)
            train_stats[f"{task}/mae"] = mae

    # ---- Add overall (aggregated) metrics ----
    if acc_vals:
        train_stats["overall_accuracy"] = float(np.mean(acc_vals))
    if mae_vals:
        train_stats["overall_mae"] = float(np.mean(mae_vals))

    return avg_loss, train_stats

# ---------------- Training loop ----------------
def run_train(cfg, model, loaders, task_info, loss_weights, device, use_tqdm):
    # torch.autograd.set_detect_anomaly(True)
    torch.autograd.set_detect_anomaly(cfg["train"].get("detect_anomaly", False))
    # --- Start timer ---
    start_time = time.time()

    # --- Model parameter summary ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    model_size_mb = total_params * 4 / (1024 ** 2)

    print("\n================ Model Summary ================")
    print(f"Total parameters   : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters   : {frozen_params:,}")
    print(f"Approx model size   : {model_size_mb:.2f} MB (float32)")
    print("==============================================\n")

    # -- Training loop ---
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"].get("weight_decay", 1e-4))
    epochs = cfg["train"]["epochs"]
    out_dir = Path(cfg["train"].get("out_dir", "runs/exp"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # best_val = float("inf")
    start_epoch = 0

    if cfg["train"].get("resume") and last_path.exists():
        state = torch.load(last_path)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state["epoch"] + 1
        print(f"==> Resumed from epoch {start_epoch}")
    
    if loaders.get("val") is None:
        raise RuntimeError("Validation loader is required for checkpoint selection and sweeps.")

    patience = cfg["train"].get("early_stopping_patience", 10)  
    early_stop_counter = 0
    best_auroc = -float("inf")

    selection_metric = cfg["train"].get("selection_metric", "loss")
    selection_mode = cfg["train"].get("selection_mode", "min")  # "min" or "max"

    if selection_mode == "min":
        best_score = float("inf")
    else:
        best_score = -float("inf")

    best_metrics = {}
    best_epoch = -1

    for epoch in range(start_epoch, epochs + 1):
        # ---- Training ----
        train_loss, train_stats = train_one_epoch(
            model, loaders["train"], optimizer, device,
            compute_multitask_loss, task_info, loss_weights,
            use_tqdm, epoch=epoch, agg=cfg["train"].get("loss_agg", "mean"),
            threshold=cfg.get("eval", {}).get("threshold", 0.5)
        )
        
        # ---- Validation ----
        overall, _ = evaluate_model(
            model, loaders["val"], device, compute_multitask_loss,
            task_info=task_info, loss_weights=loss_weights,
            threshold=cfg.get("eval", {}).get("threshold", 0.5),
            inverse_transform=False,
            agg=cfg["train"].get("loss_agg", "mean"),
            metric_level=cfg.get("eval", {}).get("metric_level", "clip"), 
            video_key="video_name",
            video_reduce_cfg=cfg["eval"].get("video_reduce", None)           
        )
        val_loss = overall["loss"]
        # val_auroc = overall.get("auroc", None)
        val_auroc = overall.get("classification", {}).get("auroc", None)
        current_score = get_nested_value(overall, selection_metric, None)

        if current_score is None:
            raise ValueError(
                f"Selection metric '{selection_metric}' not found in validation metrics. "
                f"Available top-level keys: {list(overall.keys())}"
            )

        scheduler.step()
        
        # ---- Save checkpoints ----
        if cfg["train"].get("save_checkpoints", True):
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "selection_metric": selection_metric,
                "selection_score": current_score,
            }, last_path)

        # if val_loss < best_val:
        #     best_val = val_loss
        #     early_stop_counter = 0
        #     torch.save({
        #         "epoch": epoch,
        #         "model_state": model.state_dict(),
        #         "optimizer_state": optimizer.state_dict(),
        #         "val_loss": val_loss
        #     }, best_path)
        # else:
        #     early_stop_counter += 1
        # print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")    
        # if early_stop_counter >= patience:
        #     print(f"⏹️ Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
        #     break

        improved = (
            current_score < best_score if selection_mode == "min"
            else current_score > best_score
        )

        if improved:
            best_score = current_score
            best_epoch = epoch
            best_metrics = copy.deepcopy(overall)
            early_stop_counter = 0
            if cfg["train"].get("save_checkpoints", True):
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "selection_metric": selection_metric,
                    "selection_score": current_score,
                    "best_metrics": best_metrics,
                }, best_path)
        else:
            early_stop_counter += 1

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | {selection_metric}={current_score:.4f}"
        )
        if early_stop_counter >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break


        # ---- Log to wandb ----
        # if cfg.get("wandb", {}).get("enable", False):
        #     log_dict = {
        #         "epoch": epoch,
        #         "train_loss": train_loss,
        #         "val_loss": val_loss,
        #         "lr": scheduler.get_last_lr()[0],
        #     }

        #     for k, v in overall.items():
        #         if not "/" in k:  
        #             log_dict[f"val/{k}"] = v

        #     # add overall training metrics
        #     for k, v in train_stats.items():
        #         if "overall" in k or "auroc" in k:
        #             log_dict[k] = v

        #     wandb.log(log_dict)

        if cfg.get("wandb", {}).get("enable", False):
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            }

            def flatten_dict(d, parent_key="", sep="."):
                items = {}
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.update(flatten_dict(v, new_key, sep=sep))
                    else:
                        items[new_key] = v
                return items

            flat_overall = flatten_dict(overall)
            for k, v in flat_overall.items():
                log_dict[f"val/{k}"] = v

            for k, v in train_stats.items():
                if "overall" in k or k in ["loss", "auroc"]:
                    log_dict[f"train/{k}"] = v

            # log_dict["best/selection_score"] = best_score
            wandb.log(log_dict)

    end_time = time.time()
    elapsed = end_time - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    print("\n================ Training Finished ================")
    print(f"Total training time: {h:02d}h:{m:02d}m:{s:02d}s")
    print(f"Best epoch: {best_epoch}")
    print(f"Best {selection_metric}: {best_score:.4f}")
    print("===================================================\n")
    
    return best_path, {
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "selection_score": best_score,
        "val_metrics": best_metrics,
    }