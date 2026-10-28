import torch
import torch.nn.functional as F
from videodepatchify import DePatchify
from videopatchify import VideoTokenizer

# ---------------------------------------
# Visualization helper
# ---------------------------------------
def visualize_patch(patch, C, t_p, h_p, w_p):
    """
    patch: (patch_volume,) flattened token
    Prints patch as (C, t_p, h_p, w_p)
    """
    patch_4d = patch.view(C, t_p, h_p, w_p)
    for c in range(C):
        for tt in range(t_p):
            print(f"  C={c}, T={tt}:")
            print(patch_4d[c, tt])


# ---------------------------------------
# Instantiate tokenizer & depatchify
# ---------------------------------------
tokenizer = VideoTokenizer(
    in_ch=1,
    patch=(1,2,2),    # small for visualization
    pad=(0,0,0),
    add_pos_emb=False
)

depatch = DePatchify(
    out_channels=1,
    patch=(1,2,2)
)

# ---------------------------------------
# Create a small test video
# ---------------------------------------
# Shape (1, 1, 2, 4, 4) → B,C,T,H,W
x = torch.zeros(1,2,2,4,4)
for t in range(2):
    for h in range(4):
        for w in range(4):
            x[0,0,t,h,w] = t*100 + h*10 + w

print("========== ORIGINAL INPUT ==========")
print(x)

# ---------------------------------------
# Tokenize
# ---------------------------------------
tokens, meta = tokenizer(x)

print("\n========== TOKENS ==========")
print("tokens.shape =", tokens.shape)
print("meta =", meta)

# ---------------------------------------
# Visualize each patch
# ---------------------------------------
t_bins, h_bins, w_bins = meta["bins"]
t_p, h_p, w_p = meta["patch"]
C = meta["C"]

print("\n========== PATCH VISUALIZATION ==========\n")

idx = 0
for t in range(t_bins):
    for h in range(h_bins):
        for w in range(w_bins):
            print(f"--- Token {idx} (patch coords: t={t}, h={h}, w={w}) ---")
            visualize_patch(tokens[0, idx], C, t_p, h_p, w_p)
            print()
            idx += 1

# ---------------------------------------
# Depatchify
# ---------------------------------------
x_recon = depatch(tokens, meta)

print("========== RECONSTRUCTED ==========")
print(x_recon)

print("\nReconstruction correct?", torch.allclose(x, x_recon))
