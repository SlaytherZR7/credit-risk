import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class QuantileDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins: int = 4):
        self.n_bins = int(n_bins)
        self.bin_edges_ = {}
        self.columns_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.columns_ = list(X_df.columns)
        for col in self.columns_:
            s = pd.to_numeric(X_df[col], errors='coerce')
            uniq = s.dropna().nunique()
            if uniq <= 1:
                self.bin_edges_[col] = np.array([s.min() - 1 if pd.notna(s.min()) else -1,
                                                 s.min() + 1 if pd.notna(s.min()) else 1])
                continue
            q = min(self.n_bins, max(2, uniq))
            try:
                _, bins = pd.qcut(s, q=q, retbins=True, duplicates='drop')
                bins = np.unique(bins)
            except Exception:
                vmin, vmax = s.min(), s.max()
                bins = np.linspace(vmin, vmax, num=min(self.n_bins, uniq) + 1)
            self.bin_edges_[col] = bins
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        Xt = np.zeros((len(X_df), len(self.columns_)), dtype=float)
        for i, col in enumerate(self.columns_):
            s = pd.to_numeric(X_df[col], errors='coerce')
            bins = self.bin_edges_[col]
            codes = pd.cut(s, bins=bins, labels=False, include_lowest=True)
            Xt[:, i] = pd.Series(codes).fillna(-1).astype(float).values
        return Xt
