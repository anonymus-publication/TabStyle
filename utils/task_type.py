from typing import Dict
import numpy as np
import torch

def infer_task_types(df, task_cols, exclude_cols=None) -> Dict[str, str]:
    if exclude_cols is None:
        exclude_cols = []

    types = {}
    for col in task_cols:
        if col in exclude_cols:
            continue  # skip this column

        y_np = df[col].dropna().values.astype(float)
        uniq = np.unique(y_np)
        if len(uniq) <= 2 and set(np.round(uniq).tolist()).issubset({0, 1}):
            types[col] = "binary"
        else:
            types[col] = "regression"
    return types
