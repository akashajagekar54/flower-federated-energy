import os, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset

class EnergySeries(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def _build_sequences(df, seq_len=24, horizon=1):
    feats = df[['temp','occ','hour_sin','hour_cos']].values
    target = df['load'].values
    X_list, y_list = [], []
    for i in range(seq_len, len(df) - horizon):
        X_list.append(feats[i-seq_len:i, :])
        y_list.append(target[i + horizon - 1])
    return np.stack(X_list), np.array(y_list)

def _prep(df):
    df = df.copy()
    dt = pd.to_datetime(df['timestamp'])
    hour = dt.dt.hour.values.astype(float)
    df['hour_sin'] = np.sin(2*np.pi*hour/24.0)
    df['hour_cos'] = np.cos(2*np.pi*hour/24.0)
    if 'occupancy' in df.columns and 'occ' not in df.columns:
        occ = df['occupancy'].astype(float).values
        rng = (occ.max() - occ.min()) or 1.0
        df['occ'] = (occ - occ.min())/rng
    elif 'occ' not in df.columns:
        df['occ'] = 0.0
    for col in ['temp','load']:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'")
    return df

def make_real_client_dataset(cid: int, args):
    fname = args.csv_pattern.format(cid=cid)
    path = os.path.join(args.data_root, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV for client {cid} not found at {path}")
    df = pd.read_csv(path)
    df = _prep(df)
    X, y = _build_sequences(df, seq_len=args.seq_len, horizon=1)
    return EnergySeries(X, y)
