import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

@dataclass
class EnergyConfig:
    seq_len: int = 24
    horizon: int = 1
    days: int = 120
    noise: float = 0.1
    temp_amp: float = 10.0
    base_load: float = 2.5
    noniid_strength: float = 0.3

class EnergySeries(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def _gen_client_series(cfg: EnergyConfig, seed: int = 0, client_bias: float = 0.0, client_amp_scale: float = 1.0):
    rng = np.random.default_rng(seed)
    T = cfg.days * 24
    t = np.arange(T)
    temp = cfg.temp_amp * client_amp_scale * (np.sin(2*np.pi*t/24 - np.pi/3) + 0.3*np.sin(2*np.pi*t/(24*7))) + rng.normal(0, 0.8, size=T)
    occ = np.clip((np.sin(2*np.pi*(t-8)/24) + 0.2*np.sin(2*np.pi*(t-8)/(24*7))), 0, None)
    occ = (occ > 0.2).astype(float)
    load = cfg.base_load + 0.15*temp + 1.2*occ + client_bias + rng.normal(0, cfg.noise, size=T)
    hour = (t % 24).astype(float)
    hour_sin = np.sin(2*np.pi*hour/24); hour_cos = np.cos(2*np.pi*hour/24)
    feats = np.stack([temp, occ, hour_sin, hour_cos], axis=-1)
    X_list, y_list = [], []
    for i in range(cfg.seq_len, T - cfg.horizon):
        X_list.append(feats[i-cfg.seq_len:i, :])
        y_list.append(load[i + cfg.horizon - 1])
    return np.stack(X_list), np.array(y_list)

def make_client_dataset(cid: int, cfg: EnergyConfig, base_seed: int = 42):
    import numpy as _np
    rng_bias = _np.random.default_rng(base_seed + cid)
    rng_amp = _np.random.default_rng(base_seed + 100 + cid)
    bias = rng_bias.normal(0, cfg.noniid_strength)
    amp_scale = np.clip(1.0 + rng_amp.normal(0, cfg.noniid_strength/2), 0.7, 1.3)
    X, y = _gen_client_series(cfg, seed=base_seed + 1000 + cid, client_bias=bias, client_amp_scale=amp_scale)
    return EnergySeries(X, y)

def split_train_val(dataset: EnergySeries, val_ratio: float = 0.2):
    n = len(dataset); import numpy as np
    idx = np.arange(n); split = int(n * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]
    X, y = dataset.X.numpy(), dataset.y.numpy()
    return EnergySeries(X[train_idx], y[train_idx]), EnergySeries(X[val_idx], y[val_idx])

def get_loaders(dataset: EnergySeries, batch_size: int = 64, val_ratio: float = 0.2):
    train, val = split_train_val(dataset, val_ratio)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(val, batch_size=batch_size, shuffle=False)
