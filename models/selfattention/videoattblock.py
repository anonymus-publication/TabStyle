from models.selfattention.stselftatt import SpatioTemporalSelfAttention
from models.selfattention.videodepatchify import DePatchify
from models.selfattention.videopatchify import VideoTokenizer
import torch.nn as nn
import torch
from torch.nn.functional import softmax


class VideoAttentionBlock(nn.Module):
    """
    Video attention pipeline:
    - type="none":     simple avg/flatten on (B,C,T,H,W)
    - type="spatiotemporal":
        patchify → (optional CLS) → transformer → 
        (optional depatchify) → (optional avg/flatten)
    """
    _global_print_done = False
    def __init__(self, in_channels: int, cfg: dict):
        super().__init__()

        self.debug = False
        self._printed = False
        self.print_once = True

        vcfg = cfg.get("videoattention", {})
        self.mode = vcfg.get("type", "none").lower()

        # output settings
        self.average_pool = bool(vcfg.get("average_pool", True))
        self.flatten = bool(vcfg.get("flatten", False))
        self.use_depatchify = bool(vcfg.get("depatchify", True))
        self.use_cls = bool(vcfg.get("cls_token", False))

        if self.mode == "none":
            self.out_dim = in_channels
            self.enable_attention = False
            return

        # enable attention
        self.enable_attention = True

        # patch + pad
        patch = tuple(vcfg.get("patch_size", [2, 2, 2]))
        pad   = tuple(vcfg.get("pad_size",   [0, 0, 0]))

        # tokenizer
        self.tokenizer = VideoTokenizer(
            in_ch=in_channels,
            patch=patch,
            pad=pad,
            add_pos_emb=vcfg.get("pos_emb", False),
        )

        # patch volume dimension
        t_p, h_p, w_p = patch
        token_dim = in_channels * t_p * h_p * w_p   # VERY IMPORTANT
        assert token_dim % vcfg.get("num_heads", 4) == 0

        # (optional) CLS token
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # transformer
        self.self_att = SpatioTemporalSelfAttention(
            token_dim=token_dim,
            num_heads=vcfg.get("num_heads", 4),
            num_layers=vcfg.get("num_layers", 2),
            mlp_ratio=vcfg.get("mlp_ratio", 4.0),
            dropout=vcfg.get("dropout", 0.1),
        )

        # depatchify (optional)
        if self.use_depatchify:
            self.depatch = DePatchify(out_channels=in_channels, patch=patch)

        # attention pooling (optional)
        self.use_attn_pool = bool(vcfg.get("attention_pool", False))
        if self.use_attn_pool:
            self.attn_score = nn.Linear(token_dim, 1)

        self.out_dim = in_channels  # final representation size

        


    def forward(self, feat):
        """
        feat: (B, C, T, H, W)
        """
        B, C, T, H, W = feat.shape
        debug = self.debug and (not self._printed if self.print_once else True)

        # -----------------------------------------
        # CASE 1 — NO ATTENTION
        # -----------------------------------------
        if not self.enable_attention or self.mode == "none":
            out = feat

            if self.average_pool:
                out = out.mean((2, 3, 4))    # (B,C)

            if self.flatten:
                out = out.view(B, -1)

            if debug:
                print("========== VIDEO ATTENTION (NONE) DEBUG ==========")
                print(f"Input feature map : {feat.shape}")
                print(f"Output feature map: {out.shape}")
                print("===================================================")
                self._printed = True
                VideoAttentionBlock._global_print_done = True

            return out

        # -----------------------------------------
        # CASE 2 — SPATIOTEMPORAL ATTENTION
        # -----------------------------------------
        tokens, meta = self.tokenizer(feat)    # (B, N, D)

        # Patch bins and expected tokens
        t_bins, h_bins, w_bins = meta["bins"]
        expected_tokens = t_bins * h_bins * w_bins

        # Assertion for consistency
        assert tokens.shape[1] == expected_tokens, \
            f"[ERROR] Token count mismatch: got {tokens.shape[1]} vs expected {expected_tokens}"

        # -----------------------------------------
        # DEBUG PRINT BEFORE ATTENTION
        # -----------------------------------------
        if debug:
            print("=============== VIDEO ATTENTION DEBUG START ===============")
            print(f"Backbone feature input      : {feat.shape}")          # (B,C,T,H,W)
            print(f"Patch bins (t,h,w)          : {meta['bins']}")
            print(f"Patch size (t_p,h_p,w_p)    : {meta['patch']}")
            print(f"Tokenized shape             : {tokens.shape}")       # (B,N,D)
            print(f"Expected token count        : {expected_tokens}")
            print("-----------------------------------------------------------")

        # -----------------------------------------
        # CLS TOKEN (optional)
        # -----------------------------------------
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            tokens = torch.cat([cls, tokens], dim=1)  # now (B, 1+N, D)

        # -----------------------------------------
        # TRANSFORMER SELF-ATTENTION
        # -----------------------------------------
        tokens = self.self_att(tokens)              # (B, N or N+1, D)

        if debug:
            print(f"After Transformer            : {tokens.shape}")

        # -----------------------------------------
        # CLS ONLY CASE (output CLS token)
        # -----------------------------------------
        if self.use_cls and not self.use_depatchify:
            x = tokens[:, 0]                         # (B, D)
            if debug:
                print(f"CLS output                  : {x.shape}")
                print("=============== VIDEO ATTENTION DEBUG END ===============")
                self._printed = True
                VideoAttentionBlock._global_print_done = True
            return x

        # remove CLS first
        if self.use_cls:
            tokens = tokens[:, 1:]                   # (B, N, D)

        # --------------------------------------------
        # ATTENTION POOLING CASE
        #--------------------------------------------
        if self.use_attn_pool:
            # tokens: (B, N, D)
            scores = self.attn_score(tokens)          # (B, N, 1)
            weights = torch.softmax(scores, dim=1)    # (B, N, 1)
            x = (weights * tokens).sum(dim=1)          # (B, D)

            if debug:
                print(f"Attention pooled output      : {x.shape}")
                print("=============== VIDEO ATTENTION DEBUG END ===============")
                self._printed = True
                VideoAttentionBlock._global_print_done = True

            return x

        # -----------------------------------------
        # DEPATCHIFY CASE
        # -----------------------------------------
        if self.use_depatchify:
            vol = self.depatch(tokens, meta)         # (B,C,T,H,W)

            if debug:
                print(f"Depatchified volume         : {vol.shape}")

            # ---- AVG POOL ----
            if self.average_pool:
                vol = vol.mean((2, 3, 4))            # (B,C)
                if debug:
                    print(f"After average pooling      : {vol.shape}")

            # ---- FLATTEN ----
            if self.flatten:
                vol = vol.view(B, -1)
                if debug:
                    print(f"After flatten              : {vol.shape}")
                    print("=============== VIDEO ATTENTION DEBUG END ===============")
                    self._printed = True
                    VideoAttentionBlock._global_print_done = True
            else:
                if debug:
                    print("=============== VIDEO ATTENTION DEBUG END ===============")
                    self._printed = True
                    VideoAttentionBlock._global_print_done = True

            return vol
                

        # -----------------------------------------
        # TOKEN POOL CASE (no depatchify)
        # -----------------------------------------
        x = tokens.mean(dim=1)                       # (B,D)

        if debug:
            print(f"Token pooled output          : {x.shape}")

        # Optional flatten
        if self.flatten:
            x = x.view(B, -1)
            if debug:
                print(f"After flatten               : {x.shape}")
        

        if debug:
            print("=============== VIDEO ATTENTION DEBUG END ===============")
            self._printed = True
            VideoAttentionBlock._global_print_done = True

        return x