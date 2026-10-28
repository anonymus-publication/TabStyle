import torch
import torch.nn as nn
from pathlib import Path

class MLPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        input_dim = config.get("input_dim", 256)
        hidden_dims = config.get("hidden_dims", [256, 128])
        out_dim = config.get("out_dim", 256)
        activation_name = config.get("activation", "relu").lower()
        normalization = config.get("normalization", "batchnorm").lower()
        dropout_p = config.get("dropout", 0.0)

        # debug controls
        self.debug = False
        self.print_once = True
        self._printed = False
        self.last_debug = None

        act_map = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "leakyrelu": nn.LeakyReLU(0.2, inplace=True),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        activation = act_map.get(activation_name, nn.ReLU(inplace=True))

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if normalization == "batchnorm":
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            elif normalization == "layernorm":
                layers.append(nn.LayerNorm(dims[i + 1]))
            elif normalization in ["none", None]:
                pass

            layers.append(activation)

            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))

        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.encoder = nn.Sequential(*layers)

    def reset_debug(self):
        self._printed = False
        self.last_debug = None

    def _should_debug(self):
        if not self.debug:
            return False
        if self.print_once:
            return not self._printed
        return True

    def forward(self, x):
        debug = self._should_debug()

        debug_info = {
            "input_shape": tuple(x.shape),
            "layers": []
        }

        if debug:
            print("=============== Tabular ENCODER DEBUG START ===============")
            print(f"Input tabular              : {x.shape}")

        out = x
        linear_idx = 0

        for i, layer in enumerate(self.encoder):
            in_shape = tuple(out.shape)
            out = layer(out)
            out_shape = tuple(out.shape)

            if isinstance(layer, nn.Linear):
                layer_name = f"mlp.{linear_idx}"
                linear_idx += 1

                debug_info["layers"].append({
                    "name": layer_name,
                    "type": "Linear",
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "input_shape": in_shape,
                    "output_shape": out_shape,
                })

                if debug:
                    print(f"[MLP Layer] {layer_name} (Linear [{layer.in_features} -> {layer.out_features}])            : {out_shape}")

        debug_info["final_output_shape"] = tuple(out.shape)
        self.last_debug = debug_info

        if debug:
            print(f"Final MLP output           : {out.shape}")
            print("=============== Tabular ENCODER DEBUG END ===============")
            self._printed = True

        return out