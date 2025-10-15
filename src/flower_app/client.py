import argparse, torch
from torch import nn, optim
from torch.nn.utils import prune
import flwr as fl

from src.common.model import LSTMRegressor
from src.data.synthetic_energy import EnergyConfig, make_client_dataset, get_loaders
try:
    from src.data.multibuilding_csv import make_real_client_dataset
    HAS_REAL = True
except Exception:
    HAS_REAL = False

def rmse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2).sqrt().item()

def maybe_quantize(model, do_quant):
    if not do_quant: return model
    return torch.quantization.quantize_dynamic(model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)

def maybe_prune(model, amount):
    if amount <= 0.0: return model
    params = []
    for m in model.modules():
        if isinstance(m, nn.Linear): params.append((m, 'weight'))
    if params: prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    return model

class EnergyClient(fl.client.NumPyClient):
    def __init__(self, cid:int, args):
        self.cid = cid; self.args = args
        if args.use_real and HAS_REAL:
            dataset = make_real_client_dataset(cid, args)
        else:
            cfg = EnergyConfig(seq_len=args.seq_len)
            dataset = make_client_dataset(cid, cfg, base_seed=args.seed)
        self.train_loader, self.val_loader = get_loaders(dataset, batch_size=args.batch, val_ratio=args.val_ratio)
        self.model = LSTMRegressor(input_size=4, hidden_size=args.hidden, num_layers=args.layers, dropout=args.dropout)
        self.model = maybe_prune(self.model, args.prune); self.model = maybe_quantize(self.model, args.quantize)
        self.device = torch.device(args.device); self.model.to(self.device)
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for (k, _), p in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(p, dtype=state_dict[k].dtype)
        self.model.load_state_dict(state_dict, strict=True)
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        global_params = [p.detach().clone() for p in self.model.parameters()]
        self.model.train(); opt = optim.Adam(self.model.parameters(), lr=1e-3); loss_fn = nn.MSELoss()
        steps = 0
        for _ in range(self.args.local_epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X); loss = loss_fn(pred, y)
                if self.args.mu > 0.0:
                    prox = 0.0
                    for p, g in zip(self.model.parameters(), global_params):
                        prox = prox + torch.sum((p - g.to(self.device))**2)
                    loss = loss + (self.args.mu/2.0) * prox
                opt.zero_grad(); loss.backward(); opt.step(); steps += 1
        return self.get_parameters({}), len(self.train_loader.dataset), {"num_steps": steps}
    def evaluate(self, parameters, config):
        self.set_parameters(parameters); self.model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device); p = self.model(X)
                ys.append(y); ps.append(p)
        y = torch.cat(ys); p = torch.cat(ps)
        return float(nn.MSELoss()(p, y).item()), len(self.val_loader.dataset), {"rmse": rmse(y, p)}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cid', type=int, default=0)
    parser.add_argument('--seq-len', type=int, default=24)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--local-epochs', type=int, default=2)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--server', type=str, default='127.0.0.1:8080')
    parser.add_argument('--use-real', action='store_true')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--csv-pattern', type=str, default='building_{cid}.csv')
    parser.add_argument('--prune', type=float, default=0.0)
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()
    client = EnergyClient(args.cid, args)
    fl.client.start_client(server_address=args.server, client=client.to_client())
if __name__ == "__main__":
    main()
