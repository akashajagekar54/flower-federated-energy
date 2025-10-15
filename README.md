# Flower Federated Learning: Energy Load Forecasting

## Quickstart
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# source .venv/bin/activate
pip install -r requirements.txt

# Server
python -m src.flower_app.server --strategy fedavg --rounds 30 --logdir runs/fedavg_logs

# Clients (new terminals, activate venv first)
python -m src.flower_app.client --cid 0 --mu 0.0 --server 127.0.0.1:8080
python -m src.flower_app.client --cid 1 --mu 0.0 --server 127.0.0.1:8080

# Plot
python -m src.tools.plot_logs --server-logs runs/fedavg_logs/server_log.json --labels FedAvg

