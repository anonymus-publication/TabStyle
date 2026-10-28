from pathlib import Path
import pandas as pd

def save_split_csv(loaders, save_path):
    rows = []

    for split in ["train", "val", "test"]:
        loader = loaders.get(split)
        if loader is None:
            continue

        dataset = loader.dataset
        df = dataset.tabular_df

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{split}: tabular_df is not a DataFrame")

        required_cols = ["video_name"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"{split}: missing columns {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        split_df = df[required_cols].copy()
        split_df["split"] = split
        rows.append(split_df)

    result = pd.concat(rows, ignore_index=True).drop_duplicates()

    # Check patient-level leakage
    counts = result.groupby("video_name")["split"].nunique()
    if (counts > 1).any():
        raise ValueError("Patient leakage detected across splits!")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(save_path, index=False)

    print(f"[SPLIT] Saved {len(result)} clips to {save_path}")
    print(result.groupby("split")["video_name"].nunique())