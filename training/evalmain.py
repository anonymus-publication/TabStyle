from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, mean_absolute_error
import cv2  
from utils.losses import compute_multitask_loss
from saliency.saliency import integrated_gradients_video, input_gradient_saliency, generate_saliency
from saliency.debug_sal import generate_saliency_debug, saliency_to_heatmap, overlay_heatmap
from saliency.captum import generate_saliency_captum
from saliency.saliency_integration import generate_saliency_ensemble


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

# ------------------ save eval metrices -------------------------
def save_metrics(out_dir: Path, split_name: str,
                 overall: dict[str, float],
                 per_task: dict[str, dict[str, float]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{split_name}_overall.json").open("w") as f:
        json.dump(overall, f, indent=2)
    with (out_dir / f"{split_name}_per_task.json").open("w") as f:
        json.dump(per_task, f, indent=2)
# ---------------------- Sigmoid --------------------------------
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
# ------------------- Reducer Functions -------------------------
def reduce_video_scores(vals, reduce_cfg):
    """
    vals: list[float] (probs or logits)
    reduce_cfg: dict with keys {type, ...}
    """
    rtype = reduce_cfg["type"]

    if rtype == "logit_mean":
        return float(_sigmoid(np.mean(vals)))

    if rtype == "prob_mean":
        return float(np.mean(vals))

    if rtype == "prob_max":
        return float(np.max(vals))

    if rtype == "median":
        return float(np.median(vals))

    if rtype == "topk_mean":
        k = reduce_cfg.get("topk", 0.25)
        k = max(1, int(len(vals) * k))
        topk_vals = np.sort(vals)[-k:]
        return float(np.mean(topk_vals))

    raise ValueError(f"Unknown video_reduce type: {rtype}")

def smooth_grad(model, video, n_samples=20, noise_std=0.02):
    grads = []
    for _ in range(n_samples):
        noise = torch.randn_like(video) * noise_std
        sal = input_gradient_saliency(model, video + noise)
        grads.append(sal)
    return torch.mean(torch.stack(grads), dim=0)

#------------------ Evaluating one Epoch -----------------------
@torch.no_grad()
def evaluate_model(
    model, dataloader, device, loss_fn=compute_multitask_loss,
    task_info=None, loss_weights=None, threshold=0.5,
    inverse_transform=False, reg_pipeline=None, regression_cols=None,
    agg="mean",
    metric_level="clip",
    video_key="video_name",
    video_reduce_cfg=None,
    ):
    model.eval()
    losses = []

    if video_reduce_cfg is None:
        video_reduce_cfg = {"type": "prob_mean"}

    all_targets = defaultdict(list)
    all_probs   = defaultdict(list)   # only for binary
    all_preds   = defaultdict(list)   # binary preds or regression preds

    vid_prob_buckets = defaultdict(lambda: defaultdict(list))
    vid_true_bucket = defaultdict(dict)

    for batch in dataloader:
        inputs, targets, meta = to_device_batch(batch, device, model.use_video, model.use_tabular)
        outputs = model(inputs["video"], inputs["tabular"] if model.use_tabular else None)

        # ---- Loss ----
        loss, _ = loss_fn(outputs, targets, task_info, loss_weights, agg=agg)
        losses.append(loss.item())

        # ---- Video-level checks ----
        if metric_level == "video":
            if meta is None or video_key not in meta:
                raise RuntimeError(
                    f"metric_level='video' requires meta['{video_key}'] in each batch."
                )
            video_names = meta[video_key]
            if video_names is None:
                raise RuntimeError(
                    f"meta['{video_key}'] is None. Check Dataset/Collate keys."
                )

        # ---- Collect predictions ----
        for task, y_true in targets.items():
            y_true_np = y_true.detach().cpu().numpy().ravel()
            y_out = outputs[task]

            if task_info[task] == "binary":
                if y_out.ndim > 1 and y_out.shape[-1] == 2:
                    logits = (y_out[..., 1] - y_out[..., 0]).detach().cpu().numpy().ravel()
                else:
                    logits = y_out.detach().cpu().numpy().ravel()

                prob = 1.0 / (1.0 + np.exp(-logits))

                if metric_level == "clip":
                    pred = (prob >= threshold).astype(float)
                    all_targets[task].extend(y_true_np.tolist())
                    all_probs[task].extend(prob.tolist())
                    all_preds[task].extend(pred.tolist())
                else:
                    for i, vid in enumerate(video_names):
                        t = float(y_true_np[i])
                        if task in vid_true_bucket[vid] and vid_true_bucket[vid][task] != t:
                            raise RuntimeError(f"Inconsistent label for video {vid}")
                        vid_true_bucket[vid][task] = t
                        vid_prob_buckets[vid][task].append(float(prob[i]))

            elif task_info[task] == "regression":
                pred = y_out.detach().cpu().numpy().ravel()

                if metric_level == "clip":
                    all_targets[task].extend(y_true_np.tolist())
                    all_preds[task].extend(pred.tolist())
                else:
                    for i, vid in enumerate(video_names):
                        vid_true_bucket[vid][task] = float(y_true_np[i])
                        vid_prob_buckets[vid][task].append(float(pred[i]))

            else:
                raise ValueError(f"Unknown task type for task {task}: {task_info[task]}")

    overall_stats = {
        "loss": float(np.mean(losses)) if len(losses) > 0 else float("nan")
    }
    per_task_stats = {}

    # -------- METRICS --------
    if metric_level == "video":
        for task, ttype in task_info.items():
            y_true, y_score = [], []

            for vid, task_dict in vid_prob_buckets.items():
                if task not in task_dict:
                    continue
                y_true.append(vid_true_bucket[vid][task])
                y_score.append(reduce_video_scores(task_dict[task], video_reduce_cfg))

            y_true = np.array(y_true)
            y_score = np.array(y_score)

            if len(y_true) == 0:
                per_task_stats[task] = {}
                continue

            stats = {}

            if ttype == "binary":
                pred = (y_score >= threshold).astype(float)

                # accuracy/f1 are safe if there are samples
                stats["accuracy"] = accuracy_score(y_true, pred)
                stats["f1"] = f1_score(y_true, pred, zero_division=0)

                # AUROC/AUPRC require care when only one class is present
                if len(np.unique(y_true)) > 1:
                    stats["auroc"] = roc_auc_score(y_true, y_score)
                    stats["auprc"] = average_precision_score(y_true, y_score)
                else:
                    stats["auroc"] = float("nan")
                    stats["auprc"] = float("nan")

            else:  # regression
                stats["mae"] = mean_absolute_error(y_true, y_score)

            per_task_stats[task] = stats

    else:  # clip level
        print("\n[DEBUG] Computing clip-level metrics")
        for task in all_targets:
            y_true = np.array(all_targets[task])

            if len(y_true) == 0:
                per_task_stats[task] = {}
                continue

            if task_info[task] == "binary":
                prob = np.array(all_probs[task])
                pred = np.array(all_preds[task])

                stats = {
                    "accuracy": accuracy_score(y_true, pred),
                    "f1": f1_score(y_true, pred, zero_division=0),
                }

                if len(np.unique(y_true)) > 1:
                    stats["auroc"] = roc_auc_score(y_true, prob)
                    stats["auprc"] = average_precision_score(y_true, prob)
                else:
                    stats["auroc"] = float("nan")
                    stats["auprc"] = float("nan")

                per_task_stats[task] = stats

            else:  # regression
                pred = np.array(all_preds[task])
                per_task_stats[task] = {
                    "mae": mean_absolute_error(y_true, pred)
                }

    # -------- OVERALL --------
    cls_aurocs = [v["auroc"] for v in per_task_stats.values() if "auroc" in v and not np.isnan(v["auroc"])]
    cls_auprcs = [v["auprc"] for v in per_task_stats.values() if "auprc" in v and not np.isnan(v["auprc"])]
    cls_accs   = [v["accuracy"] for v in per_task_stats.values() if "accuracy" in v and not np.isnan(v["accuracy"])]
    cls_f1s    = [v["f1"] for v in per_task_stats.values() if "f1" in v and not np.isnan(v["f1"])]
    reg_maes   = [v["mae"] for v in per_task_stats.values() if "mae" in v and not np.isnan(v["mae"])]

    overall_stats["classification"] = {
        "auroc": float(np.mean(cls_aurocs)) if cls_aurocs else float("nan"),
        "auprc": float(np.mean(cls_auprcs)) if cls_auprcs else float("nan"),
        "accuracy": float(np.mean(cls_accs)) if cls_accs else float("nan"),
        "f1": float(np.mean(cls_f1s)) if cls_f1s else float("nan"),
    }

    overall_stats["regression"] = {
        "mae": float(np.mean(reg_maes)) if reg_maes else float("nan")
    }

    return overall_stats, per_task_stats

# ------------------ main Function ------------------------------
def run_eval(cfg, model, loader, task_info, loss_weights, device, split: str, inverse_transform=False ,reg_pipeline=None, regression_cols=None):
    out_dir = Path(cfg["train"].get("out_dir", "runs/exp"))
    ckpt = cfg.get("eval", {}).get("checkpoint", str(out_dir / "best.pt"))
    ckpt_path = Path(ckpt)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"⚠️ Checkpoint not found at {ckpt_path}. Evaluating current weights.")
    overall, per_task = evaluate_model(
        model, loader, device, compute_multitask_loss, task_info=task_info, loss_weights=loss_weights, threshold=cfg.get("eval", {}).get("threshold", 0.5), 
        inverse_transform=inverse_transform, reg_pipeline=reg_pipeline, regression_cols=regression_cols, metric_level=cfg.get("eval", {}).get("metric_level", "clip"), video_key="video_name",
        video_reduce_cfg=cfg["eval"].get("video_reduce", None))
    # print(f"Metrices are reported in video level")
    print(f"\n{split.upper()} OVERALL:", overall)
    print(f"\n{split.upper()} PER-TASK:", json.dumps(per_task, indent=2))

    metrics_dir = out_dir / "metrics"
    save_metrics(metrics_dir, split, overall, per_task)

    # --- Saliency: separate pass, no no_grad wrapper ---
    print(f"[Config] saliency section: {cfg.get('saliency', 'NOT FOUND')}")
    if cfg["saliency"].get("enable", False):
        print(f"\n[Saliency] Generating SmoothIG maps for {split} set...")
        # generate_saliency(
        #     model=model,
        #     dataloader=loader,
        #     device=device,
        #     task_name=cfg["saliency"].get("saliency_task", "Rythm_M24"),
        #     saliency_out_dir=out_dir / "saliency" / split,
        #     max_batches=cfg["saliency"].get("saliency_batches", None),
        #     steps=cfg["saliency"].get("saliency_steps", 8),
        #     baseline=cfg["saliency"].get("saliency_baseline", "blur"),
        #     blur_kernel=tuple(cfg["saliency"].get("blur_kernel", (1, 11, 11))),
        #     blur_sigma=cfg["saliency"].get("blur_sigma", 3),
        #     chunk_size=cfg["saliency"].get("saliency_chunk_size", 1),
        #     normalize=cfg["saliency"].get("saliency_normalize", "per_video"),
        #     ref_idx=cfg["saliency"].get("saliency_ref_idx", 0),
        #     temporal_sigma=cfg["saliency"].get("saliency_temporal_sigma", 1.0),
        #     stabilize=cfg["saliency"].get("saliency_stabilize", True),
        # )

        # generate_saliency_debug(
        #     model=model,
        #     dataloader=loader,
        #     device=device,
        #     task_name=cfg["saliency"].get("saliency_task", "Rythm_M24"),
        #     saliency_out_dir=out_dir / "saliency" / split,
        #     max_batches=cfg["saliency"].get("saliency_batches", None),
        #     method=cfg["saliency"].get("method", "ig"),
        #     steps=cfg["saliency"].get("saliency_steps", 8),
        #     baseline=cfg["saliency"].get("saliency_baseline", "blur"),
        #     blur_kernel=tuple(cfg["saliency"].get("blur_kernel", (1, 11, 11))),
        #     blur_sigma=cfg["saliency"].get("blur_sigma", 3),
        #     chunk_size=cfg["saliency"].get("saliency_chunk_size", 1),
        #     smooth_n_samples=cfg["saliency"].get("smooth_n_samples", 25),
        #     smooth_noise_level=cfg["saliency"].get("smooth_noise_level", 0.10),
        #     normalize=cfg["saliency"].get("saliency_normalize", "per_video"),
        #     ref_idx=cfg["saliency"].get("saliency_ref_idx", 0),
        #     temporal_sigma=cfg["saliency"].get("saliency_temporal_sigma", 1.0),
        #     stabilize=cfg["saliency"].get("saliency_stabilize", True),
        #     auto_check_flow_sign=True,
        #     save_flow_sign_debug=True,
        # )

        generate_saliency_captum(
            model=model,
            dataloader=loader,
            device=device,
            task_name=cfg["saliency"].get("saliency_task", "Rythm_M24"),
            saliency_out_dir=out_dir / "saliency" / split,
            max_batches=cfg["saliency"].get("saliency_batches", None),
            method=cfg["saliency"].get("method", "IntegratedGradients"),
            steps=cfg["saliency"].get("saliency_steps", 8),
            baseline=cfg["saliency"].get("saliency_baseline", "blur"),
            blur_kernel=tuple(cfg["saliency"].get("blur_kernel", (1, 11, 11))),
            smooth_n_samples=cfg["saliency"].get("smooth_n_samples", 25),
            smooth_noise_level=cfg["saliency"].get("smooth_noise_level", 0.10),
            normalize=cfg["saliency"].get("saliency_normalize", "per_video"),
            ref_idx=cfg["saliency"].get("saliency_ref_idx", 0),
            temporal_sigma=cfg["saliency"].get("saliency_temporal_sigma", 1.0),
            threshold=cfg.get("eval", {}).get("threshold", 0.5),
            stabilize=cfg["saliency"].get("saliency_stabilize", True),
        )
        
        # # Extract saliency config
        # saliency_cfg = cfg.get("saliency", {})

        # generate_saliency_ensemble(
        #     model=model,
        #     dataloader=loader,
        #     device=device,
        #     task_name=saliency_cfg.get("saliency_task", "Rythm_M24"),
        #     saliency_out_dir=out_dir / "saliency" / split,
        #     max_batches=saliency_cfg.get("saliency_batches", None),
        #     # ===== ENSEMBLE CONFIGURATION =====
        #     methods=saliency_cfg.get("methods", ["ig", "attention", "occlusion", "tabular_aware"]),
        #     ensemble_weights=saliency_cfg.get("ensemble_weights", None),  # Defaults to equal
        #     # ===== IG PARAMETERS =====
        #     ig_steps=saliency_cfg.get("ig_steps", 50),
        #     ig_baseline=saliency_cfg.get("ig_baseline", "blur"),
        #     ig_blur_kernel=tuple(saliency_cfg.get("blur_kernel", [1, 11, 11])),
        #     # ===== OCCLUSION PARAMETERS =====
        #     occlusion_patch_size=tuple(saliency_cfg.get("occlusion_patch_size", [1, 8, 8])),
        #     occlusion_stride=tuple(saliency_cfg.get("occlusion_stride", [1, 4, 4])),
        #     # ===== NORMALIZATION & SMOOTHING =====
        #     normalize=saliency_cfg.get("saliency_normalize", "per_video"),
        #     temporal_sigma=saliency_cfg.get("saliency_temporal_sigma", 1.0),
        #     apply_stabilization=saliency_cfg.get("saliency_stabilize", True),
        #     # ===== VISUALIZATION =====
        #     heatmap_alpha=saliency_cfg.get("heatmap_alpha", 0.4),
        #     save_individual_maps=saliency_cfg.get("save_individual_maps", True),
        #     save_ensemble_only=saliency_cfg.get("save_ensemble_only", False),
        #     threshold=cfg.get("eval", {}).get("threshold", 0.5),
        # )
