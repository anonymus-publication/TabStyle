import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class TabularNormalizer:
    def __init__(self, regression_cols, binary_cols):
        self.regression_cols = regression_cols
        self.binary_cols = binary_cols
        self.scaler = StandardScaler()

    def fit(self, df):
        self.scaler.fit(df[self.regression_cols])

    def transform(self, df_row):
        reg_data = self.scaler.transform([df_row[self.regression_cols].values])[0]  # shape: (D1,)
        bin_data = df_row[self.binary_cols].values.astype(float)                    # shape: (D2,)
        combined = np.concatenate([reg_data, bin_data])
        return torch.tensor(combined).float()  # shape: (D1+D2,)
