import torch
import torch.nn as nn
from models.backbone.convblock import Conv2Plus1DBlock
from models.backbone.basicblock import BasicBlock
from utils.attention import AttentionPooling
from models.selfattention.videoattblock import VideoAttentionBlock

class Custom3DNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.out_channels = self.config["layers"][-1]["out_channels"]
        self.stem = Conv2Plus1DBlock(**config["stem"])
        self.layers = nn.ModuleDict()
        self.attentions = nn.ModuleDict()

        self.debug = False
        self.print_once = True
        self._printed = False
        self.last_debug = None

        base_att_cfg = config.get("videoattention", {})
        
        # Create layers + optional attention
        for layer_cfg in config["layers"]:
            name = layer_cfg["name"]

            # ---- build conv layer once ----
            self.layers[name] = self._make_layer(layer_cfg)

            # ---- build attention ----
            att_cfg = layer_cfg.get("attention", None)

            if att_cfg is None or att_cfg.get("type", "none") == "none":
                self.attentions[name] = nn.Identity()
            else:
                merged_cfg = dict(base_att_cfg)
                merged_cfg.update(att_cfg)
                self.attentions[name] = VideoAttentionBlock(
                    in_channels=layer_cfg["out_channels"],
                    cfg={"videoattention": merged_cfg}
                )

        self.out_dim = config["layers"][-1]["out_channels"]
        self._printed_shape = False

    def reset_debug(self):
        self._printed = False
        self._printed_shape = False
        self.last_debug = None

        for module in self.attentions.values():
            if hasattr(module, "reset_debug"):
                module.reset_debug()
            elif hasattr(module, "_printed"):
                module._printed = False

    def should_debug(self):
        if not self.debug:
            return False
        if self.print_once:
            return not self._printed
        return True

    def _make_layer(self, cfg):
        in_ch = cfg["in_channels"]
        out_ch = cfg["out_channels"]
        mid_ch = cfg["mid_channels"]
        num_blocks = cfg["num_blocks"]
        stride = tuple(cfg.get("stride", [1, 1, 1]))
        use_temporal_conv = cfg.get("use_temporal_conv", True)
        spatial_kernel_size = tuple(cfg.get("spatial_kernel_size", [1, 3, 3]))
        temporal_kernel_size = tuple(cfg.get("temporal_kernel_size", [3, 1, 1]))
        dropout = cfg.get("dropout", 0.0) 

        # Downsample the x to prepare to addition to f(x) if stride changes or channels change
        downsample = None
        if stride != (1, 1, 1) or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm3d(out_ch),
                nn.GroupNorm(num_groups=8, num_channels=out_ch)
                # nn.InstanceNorm3d(out_ch, affine=True),
            )

        blocks = []
        for i in range(num_blocks):
            block_cfg = {
                "in_channels": in_ch if i == 0 else out_ch,
                "mid_channels": mid_ch,
                "out_channels": out_ch,
                "stride": stride if i == 0 else (1, 1, 1),
                "padding": (1, 1, 1),
                "downsample": downsample if i == 0 else None,
                "use_temporal_conv": use_temporal_conv,
                "spatial_kernel_size": spatial_kernel_size,
                "temporal_kernel_size": temporal_kernel_size,
                "dropout": dropout,  
            }
            blocks.append(BasicBlock(**block_cfg))
        return nn.Sequential(*blocks)
    
    
    def forward(self, video):
        debug = self.should_debug()

        debug_info = {
            "input_shape": tuple(video.shape),
            "stem_output_shape": None,
            "layers": []
        }


        x = self.stem(video)
        debug_info["stem_output_shape"] = tuple(x.shape)


        for layer_cfg in self.config["layers"]:
            name = layer_cfg["name"]

            x = self.layers[name](x)
            conv_shape = tuple(x.shape)

            x = self.attentions[name](x)
            att_shape = tuple(x.shape)

            debug_info["layers"].append({
                "name": name,
                "conv_output_shape": conv_shape,
                "attention_type": self.attentions[name].__class__.__name__,
                "attention_output_shape": att_shape,
            })

        if debug:
            print("=============== VIDEO ENCODER DEBUG START ===============")
            print(f"Input video                 : {video.shape}")
            print(f"After stem                  : {x.shape}")
            for layer_info in debug_info["layers"]:
                print(f"After conv {layer_info['name']:<12}   : {layer_info['conv_output_shape']}")
                print(f"After attention {layer_info['name']:<7}   : {layer_info['attention_output_shape']} ({layer_info['attention_type']})")
            print(f"Backbone final output       : {x.shape}")
            print("=============== VIDEO ENCODER DEBUG END ===============")
            self._printed = True
        
        self.last_debug = debug_info

        # if not self._printed_shape:
        #     print("[DEBUG] Backbone output shape:", x.shape)
        #     self._printed_shape = True
        
        if self.out_channels is None:
            self.out_channels = x.shape[1]
            
        return x