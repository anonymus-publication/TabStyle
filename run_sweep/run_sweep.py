import copy
from pathlib import Path
import yaml
import wandb

from train import main_from_config   # adjust import to your file name


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_nested_value(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    d = cfg

    for k in keys[:-1]:
        if isinstance(d, list):
            d = d[int(k)]
        else:
            d = d[k]

    if isinstance(d, list):
        d[int(keys[-1])] = value
    else:
        d[keys[-1]] = value


def apply_overrides(cfg, overrides):
    for k, v in overrides.items():
        set_nested_value(cfg, k, v)
    return cfg


def sweep_run():
    wandb.init()

    base_cfg = load_yaml(Path(__file__).resolve().parent / "config" / "config.yaml")
    cfg = copy.deepcopy(base_cfg)

    apply_overrides(cfg, dict(wandb.config))

    run_name = wandb.run.name or "sweep_run"
    if cfg.get("wandb", {}).get("enable", False):
        cfg["wandb"]["name"] = run_name

    cfg["train"]["out_dir"] = f"runs/sweeps/{run_name}"
    cfg["train"]["resume"] = False
    cfg["train"]["save_checkpoints"] = False

    results = main_from_config(cfg)

    if results is not None:
        best_info = results.get("best_info", {})
        wandb.summary["best_checkpoint"] = results.get("best_checkpoint", "")
        wandb.summary["best_epoch"] = best_info.get("best_epoch", -1)
        wandb.summary["selection_metric"] = best_info.get("selection_metric", "")
        wandb.summary["selection_score"] = best_info.get("selection_score", None)


if __name__ == "__main__":
    sweep_run()