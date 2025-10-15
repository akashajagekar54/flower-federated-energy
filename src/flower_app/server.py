import argparse, json, os
import flwr as fl
from flwr.common import parameters_to_ndarrays
from src.flower_app.fednova import FedNova

def estimate_comm_bytes(parameters) -> int:
    arrs = parameters_to_ndarrays(parameters)
    return sum(a.size * a.itemsize for a in arrs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--min-fit-clients', type=int, default=2)
    parser.add_argument('--min-available-clients', type=int, default=2)
    parser.add_argument('--fraction-fit', type=float, default=1.0)
    parser.add_argument('--strategy', type=str, default='fedavg', choices=['fedavg','fednova'])
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--logdir', type=str, default='runs/server_logs')
    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, 'server_log.json')
    log = {'round': [], 'comm_bytes': [], 'round_eval': [], 'round_rmse': []}
    if args.strategy == 'fedavg':
        base_strategy = fl.server.strategy.FedAvg(fraction_fit=args.fraction_fit, min_fit_clients=args.min_fit_clients, min_available_clients=args.min_available_clients)
    else:
        base_strategy = FedNova(fraction_fit=args.fraction_fit, min_fit_clients=args.min_fit_clients, min_available_clients=args.min_available_clients, lr=args.lr)
    class LoggingStrategy(type(base_strategy)):
        def aggregate_fit(self, server_round, results, failures):
            new_params, metrics = super().aggregate_fit(server_round, results, failures)
            if results:
                try:
                    downlink = estimate_comm_bytes(results[0][1].parameters) * len(results)
                except Exception:
                    downlink = 0
                uplink = sum(estimate_comm_bytes(fr.parameters) for _, fr in results)
                log['round'].append(server_round); log['comm_bytes'].append(downlink + uplink)
                with open(log_path, 'w') as f: json.dump(log, f, indent=2)
            return new_params, metrics
        def aggregate_evaluate(self, server_round, results, failures):
            if results:
                rmses = [float(ev.metrics.get("rmse")) for _, ev in results if ev.metrics and "rmse" in ev.metrics]
                if rmses:
                    log['round_eval'].append(server_round); log['round_rmse'].append(sum(rmses)/len(rmses))
                    with open(log_path, 'w') as f: json.dump(log, f, indent=2)
            return super().aggregate_evaluate(server_round, results, failures)
    strategy = LoggingStrategy()
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=args.rounds))
if __name__ == "__main__":
    main()
