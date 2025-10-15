from typing import List, Tuple, Dict, Optional
import numpy as np
import flwr as fl
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters

class FedNova(fl.server.strategy.FedAvg):
    def __init__(self, lr: float = 1.0, **kwargs):
        super().__init__(**kwargs); self.lr = lr
    def aggregate_fit(self, server_round: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], failures: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]]):
        if not results: return None, {}
        initial = None; deltas_norm=[]; sizes=[]
        for _, fit_res in results:
            w = parameters_to_ndarrays(fit_res.parameters)
            if initial is None: initial = [arr.copy() for arr in w]
            delta = [wi - w0 for wi, w0 in zip(w, initial)]
            steps = fit_res.metrics.get("num_steps", fit_res.num_examples); steps = max(float(steps), 1.0)
            deltas_norm.append([d/steps for d in delta]); sizes.append(fit_res.num_examples)
        total = float(sum(sizes)); weights = [s/total for s in sizes]
        agg = []
        for params in zip(*deltas_norm):
            stacked = np.stack(params, axis=0)
            w = np.array(weights, dtype=stacked.dtype).reshape(-1, *([1]*(stacked.ndim-1)))
            agg.append((stacked*w).sum(axis=0))
        new_weights = [w0 + self.lr*g for w0, g in zip(initial, agg)]
        return ndarrays_to_parameters(new_weights), {}
