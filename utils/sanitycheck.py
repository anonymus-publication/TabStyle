import torch
import torch.nn as nn
from models.selfattention.videoattblock import VideoAttentionBlock


def tensor_stats(x, name="tensor"):
    if not torch.is_tensor(x):
        print(f"{name}: not a tensor")
        return
    print(
        f"{name}: shape={tuple(x.shape)} | "
        f"mean={x.mean().item():.4f} | std={x.std().item():.4f} | "
        f"min={x.min().item():.4f} | max={x.max().item():.4f}"
    )

def register_affine_hooks(model, hook_storage):
    """
    Register forward hooks on possible affine/FiLM modulation layers.
    This is generic: it tries to find modules whose names suggest
    modulation / gamma-beta / FiLM / affine conditioning.
    """
    keywords = ["affine", "film", "gamma", "beta", "modulation", "condition"]

    for name, module in model.named_modules():
        lname = name.lower()

        if any(k in lname for k in keywords):
            def hook_fn(mod, inp, out, module_name=name):
                print(f"\n[Affine Hook] {module_name}")
                if isinstance(inp, tuple):
                    for i, x in enumerate(inp):
                        if torch.is_tensor(x):
                            tensor_stats(x, f"{module_name}.input[{i}]")
                else:
                    if torch.is_tensor(inp):
                        tensor_stats(inp, f"{module_name}.input")

                if torch.is_tensor(out):
                    tensor_stats(out, f"{module_name}.output")
                elif isinstance(out, (tuple, list)):
                    for i, x in enumerate(out):
                        if torch.is_tensor(x):
                            tensor_stats(x, f"{module_name}.output[{i}]")

            h = module.register_forward_hook(hook_fn)
            hook_storage.append(h)
            print(f"[Check] Registered affine/modulation hook on: {name}")

def register_tabular_hooks(tabular_encoder, hook_storage, detailed=True):
    """
    detailed=True  -> hook Linear + norm + activation + dropout
    detailed=False -> hook only Linear layers
    """
    if detailed:
        allowed = (
            nn.Linear,
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.Dropout,
            nn.ReLU,
            nn.LeakyReLU,
        )
    else:
        allowed = (nn.Linear,)

    for name, module in tabular_encoder.named_modules():
        if isinstance(module, allowed):
            def hook_fn(mod, inp, out, module_name=name):
                info = mod.__class__.__name__

                if isinstance(mod, nn.Linear):
                    info += f" [{mod.in_features} -> {mod.out_features}]"

                print(f"[Tabular Layer] {module_name} ({info})")

                if torch.is_tensor(out):
                    if detailed:
                        tensor_stats(out, f"{module_name}.output")
                    else:
                        print(f"[Check] {module_name}.output shape = {tuple(out.shape)}")

            h = module.register_forward_hook(hook_fn)
            hook_storage.append(h)

def register_video_hooks(video_encoder, hook_storage, detailed=True):
    """
    Register forward hooks to inspect tensor flow in the video encoder.
    Set detailed=False if you only want high-level outputs.
    """

    high_level_names = {"stem", "layers", "layer1", "layer2", "layer3", "layer4"}

    detailed_allowed = (
        nn.Conv3d,
        nn.BatchNorm3d,
        nn.InstanceNorm3d,
        nn.ReLU,
        nn.LeakyReLU,
        nn.MaxPool3d,
        nn.AdaptiveAvgPool3d,
        nn.Dropout,
    )

    for name, module in video_encoder.named_modules():
        module_type = module.__class__.__name__

        should_hook = False

        if detailed:
            if isinstance(module, detailed_allowed):
                should_hook = True

            # also include your custom blocks by class name
            if module_type in ["BasicBlock", "Conv2Plus1D", "R2Plus1dStem", "VideoResNet", "Custom3DNet"]:
                should_hook = True
        else:
            # high-level only
            if name in high_level_names or module_type in ["BasicBlock"]:
                should_hook = True

        if should_hook:
            def hook_fn(mod, inp, out, module_name=name, module_type=module_type):
                print(f"[Video Layer] {module_name} ({module_type})")

                if isinstance(mod, nn.Conv3d):
                    print(
                        f"  kernel={mod.kernel_size}, stride={mod.stride}, "
                        f"padding={mod.padding}, out_channels={mod.out_channels}"
                    )

                if torch.is_tensor(out):
                    tensor_stats(out, f"{module_name}.output")
                elif isinstance(out, (tuple, list)):
                    for i, x in enumerate(out):
                        if torch.is_tensor(x):
                            tensor_stats(x, f"{module_name}.output[{i}]")

            h = module.register_forward_hook(hook_fn)
            hook_storage.append(h)


def sanity_check(model, loaders, device, max_tasks_to_print=None):
    print("\n================ SANITY CHECK START ================")

    model.eval()
    hook_storage = []

    with torch.no_grad():
        # --------------------------------------------------
        # 1) Model overview
        # --------------------------------------------------
        print("\n================== MODEL OVERVIEW ==================")
        print(f"Model classes         : {model.__class__.__name__}")
        if hasattr(model, "model_name"):
            print(f"Model name            : {model.model_name}")

        print("\nAvailable top-level modules:")
        for attr in [
            "video_encoder",
            "tabular_encoder",
            "modulation",
            "video_att",
            "heads",
        ]:
            if hasattr(model, attr):
                print(f"  - {attr}: Found")
            else:
                print(f"  - {attr}: Not found")

        # --------------------------------------------------
        # 5) Load one mini-batch
        # --------------------------------------------------
        print("\n================== INPUT BATCH ==================")
        sample_batch = next(iter(loaders["train"]))

        if "frames" not in sample_batch:
            raise KeyError("sample_batch must contain key 'frames'")
        if "tabular" not in sample_batch:
            raise KeyError("sample_batch must contain key 'tabular'")

        video_batch = sample_batch["frames"].to(device)
        tab_batch = sample_batch["tabular"].to(device)

        tensor_stats(video_batch, "Video Input Batch")
        tensor_stats(tab_batch, "Tabular Input Batch")

        # also print targets if available
        for k, v in sample_batch.items():
            if k not in ["frames", "tabular"] and torch.is_tensor(v):
                tensor_stats(v, f"target[{k}]")

        # --------------------------------------------------
        # 3) Video encoder branch only
        # --------------------------------------------------
        if hasattr(model, "video_encoder") and model.video_encoder is not None:
            model.video_encoder.debug = True
            model.video_encoder.print_once = True
            model.video_encoder.reset_debug()

            if hasattr(model.video_encoder, "attentions"):
                for _, module in model.video_encoder.attentions.items():
                    if hasattr(module, "debug"):
                        module.debug = True
                    if hasattr(module, "print_once"):
                        module.print_once = True
                    if hasattr(module, "reset_debug"):
                        module.reset_debug()

            feat = model.video_encoder(video_batch)

        # --------------------------------------------------
        # 4) Tabular encoder branch only
        # --------------------------------------------------
        if hasattr(model, "tabular_encoder") and model.tabular_encoder is not None:
            model.tabular_encoder.debug = True
            model.tabular_encoder.print_once = True
            model.tabular_encoder.reset_debug()
            

        # --------------------------------------------------
        # 4) MODULATION BLOCK DEBUG
        # --------------------------------------------------
        if hasattr(model, "modulation") and model.modulation is not None:
            model.modulation.debug = True
            model.modulation._printed = False

        # --------------------------------------------------
        # 3) Detect global attention
        # --------------------------------------------------
        for name, module in model.named_modules():
            if isinstance(module, VideoAttentionBlock):
                module.debug = True
                module._printed = False
        
        # --------------------------------------------------
        # 3) Prediction heads
        # --------------------------------------------------
        # print("=============== Prediction Head DEBUG START ===============")
        # if hasattr(model, "heads") and model.heads is not None:
        #     for task_name, head in model.heads.items():
        #         if hasattr(head, "debug"):
        #             head.debug = True
        #         if hasattr(head, "print_once"):
        #             head.print_once = True
        #         if hasattr(head, "reset_debug"):
        #             head.reset_debug()


        # --------------------------------------------------
        # 4) Output Statics
        # --------------------------------------------------
        outputs = model(video_batch, tab_batch)
        print("\n================ OUTPUT STATISTICS ================")
        if isinstance(outputs, dict):
            for task_name, y_out in outputs.items():
                tensor_stats(y_out, f"[{task_name}]")
        else:
            tensor_stats(outputs, "model_output")


        # --------------------------------------------------
        # 5) Check for NaNs / Infs in parameters
        # --------------------------------------------------
        print("\n================ PARAMETER HEALTH CHECK ================")
        nan_found = False
        inf_found = False
        for name, p in model.named_parameters():
            if p is None:
                continue
            if torch.isnan(p).any():
                print(f"[WARNING] NaN found in parameter: {name}")
                nan_found = True
            if torch.isinf(p).any():
                print(f"[WARNING] Inf found in parameter: {name}")
                inf_found = True

        if not nan_found and not inf_found:
            print("[Check] No NaN/Inf found in model parameters.")

    # remove hooks
    for h in hook_storage:
        h.remove()

    print("\n================ SANITY CHECK END ================\n")