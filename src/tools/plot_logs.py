import argparse, json, os
import matplotlib.pyplot as plt

def load_log(path):
    if not os.path.exists(path): return {}
    with open(path, 'r') as f: return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-logs', type=str, nargs='+', default=['runs/server_logs/server_log.json'])
    parser.add_argument('--labels', type=str, nargs='*', default=None)
    parser.add_argument('--out', type=str, default='runs/plots')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Communication per round
    plt.figure()
    for i, path in enumerate(args.server_logs):
        s = load_log(path); rounds = s.get('round', []); comm = s.get('comm_bytes', [])
        if not rounds or not comm: continue
        label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(os.path.dirname(path))
        plt.plot(rounds, [c/1e6 for c in comm], label=label)
    plt.xlabel('Round'); plt.ylabel('Communication (MB)'); plt.title('Communication per Round')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.out, 'comm_per_round.png')); plt.close()

    # Mean RMSE per round
    plt.figure()
    for i, path in enumerate(args.server_logs):
        s = load_log(path); rounds = s.get('round_eval', []); rmses = s.get('round_rmse', [])
        if not rounds or not rmses: continue
        label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(os.path.dirname(path))
        plt.plot(rounds, rmses, label=label)
    plt.xlabel('Round'); plt.ylabel('Mean Client RMSE'); plt.title('Evaluation RMSE per Round')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.out, 'rmse_per_round.png')); plt.close()

if __name__ == '__main__':
    main()
