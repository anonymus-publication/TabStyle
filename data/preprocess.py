import json
import random
import pandas as pd
from pathlib import Path


def load_or_create_balanced_subset(train_df, config):
    """
    Balances ONLY the TRAIN subset (never val/test).
    Caches the result safely, invalidating cache if:
      - train split changed
      - random seed changed
      - mode changed
    """

    mode       = config["balance"]["mode"]
    cache_path = Path(config["balance"]["cache"])
    seed       = config.get("seed", 42)

    video_id_col = config["data"]["task_cols"][0]
    label_col    = config["data"]["task_cols"][1]

    # --------------------------------------------------------
    # If cache exists → load and verify
    # --------------------------------------------------------
    if cache_path.exists():
        print(f"🔁 Loading cached balanced subset: {cache_path}")
        subset = json.loads(cache_path.read_text())

        # Check if compatibility holds
        current_videos = set(train_df["video_name"].unique())
        cached_videos  = set(subset["train_split_videos"])

        cache_seed = subset.get("seed", None)
        cache_mode = subset.get("mode", None)

        # Invalidate cache if:
        #   - split changed
        #   - seed changed
        #   - balancing mode changed
        if (
            current_videos != cached_videos
            or cache_seed != seed
            or cache_mode != mode
        ):
            print("⚠️ Cached subset does not match current train split/seed/mode!")
            print("   → Recomputing balanced subset...")
            cache_path.unlink()  # remove invalid cache
        else:
            print("✓ Cache is valid — using saved balanced subset.")
            return apply_subset(train_df, subset, video_id_col)

    # --------------------------------------------------------
    # Cache invalid or does not exist → compute new subset
    # --------------------------------------------------------
    if mode == "none":
        subset = {
            "mode": "none",
            "seed": seed,
            "train_split_videos": train_df["video_name"].unique().tolist(),
            "selected_video_names": train_df["video_name"].unique().tolist(),
            "selected_video_ids": train_df[video_id_col].tolist(),
        }

    elif mode == "video_balancing":
        subset = balance_by_video(train_df, video_id_col, label_col, seed)

    elif mode == "clip_balancing":
        subset = balance_by_clip(train_df, video_id_col, label_col, seed)

    else:
        raise ValueError(f"Unknown balancing mode: {mode}")

    # --------------------------------------------------------
    # Save with metadata for validation next time
    # --------------------------------------------------------
    subset["seed"] = seed
    subset["mode"] = mode
    subset["train_split_videos"] = train_df["video_name"].unique().tolist()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(subset, indent=2))
    print(f"💾 Saved balanced subset to {cache_path}")

    return apply_subset(train_df, subset, video_id_col)



# ===========================================================
# BALANCING STRATEGIES
# ===========================================================

def balance_by_video(df, video_id_col, label_col, seed):
    """
    Select equal number of positive/negative *videos* (video_name),
    keep all their clips (video_id).
    """
    random.seed(seed)

    pos_videos = df[df[label_col] == 1]["video_name"].unique().tolist()
    neg_videos = df[df[label_col] == 0]["video_name"].unique().tolist()

    n = min(len(pos_videos), len(neg_videos))

    selected_pos = random.sample(pos_videos, n)
    selected_neg = random.sample(neg_videos, n)
    selected = selected_pos + selected_neg

    # Keep all clips belonging to those videos
    selected_clips = df[df["video_name"].isin(selected)][video_id_col].tolist()

    return {
        "mode": "video_balancing",
        "selected_video_names": selected,
        "selected_video_ids": selected_clips,
    }



def balance_by_clip(df, video_id_col, label_col, seed):
    """
    Balance based on clip samples directly.
    """
    random.seed(seed)

    pos_clips = df[df[label_col] == 1][video_id_col].tolist()
    neg_clips = df[df[label_col] == 0][video_id_col].tolist()

    n = min(len(pos_clips), len(neg_clips))

    selected = random.sample(pos_clips, n) + random.sample(neg_clips, n)
    selected_videos = df[df[video_id_col].isin(selected)]["video_name"].unique().tolist()

    return {
        "mode": "clip_balancing",
        "selected_video_names": selected_videos,
        "selected_video_ids": selected,
    }



# ===========================================================
# APPLY SUBSET TO THE DATAFRAME
# ===========================================================

def apply_subset(df, subset, video_id_col):
    selected = set(subset["selected_video_ids"])
    df_filtered = df[df[video_id_col].isin(selected)].reset_index(drop=True)
    print(f"📌 After balancing: {len(df_filtered)} rows")
    return df_filtered



def balance_videos_globally(df, target_col, seed=42):
    """
    Balance dataset at VIDEO LEVEL before splitting.
    Keeps full clips per video.
    """
    video_labels = (
        df.groupby("video_name")[target_col]
        .first()
        .reset_index()
    )

    counts = video_labels[target_col].value_counts()
    min_count = counts.min()

    balanced_videos = (
        video_labels
        .groupby(target_col)
        .sample(n=min_count, random_state=seed)
    )["video_name"]

    print("🔥 Global video-level balancing Applied")

    return df[df["video_name"].isin(balanced_videos)].reset_index(drop=True)