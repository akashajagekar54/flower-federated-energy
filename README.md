# Flower Federated Learning: Energy Load Forecasting (Final)
Clean, human‑commented Flower + PyTorch demo with FedAvg, FedProx, and a FedNova‑like strategy.
Includes synthetic and real multi-building loaders, pruning/quantization, and plots.

Developed as part of Cornell University Systems Engineering MS work (2024–2025), and aligned with
remote sensing / federated learning directions of the NICE Lab (Boise State University).

## Quickstart
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt

# Server
python -m src.flower_app.server --strategy fedavg --rounds 30 --logdir runs/fedavg_logs

# Clients (new terminals, activate venv first)
python -m src.flower_app.client --cid 0 --mu 0.0 --server 127.0.0.1:8080
python -m src.flower_app.client --cid 1 --mu 0.0 --server 127.0.0.1:8080

# Plot
python -m src.tools.plot_logs --server-logs runs/fedavg_logs/server_log.json --labels FedAvg

## GitHub upload
git init
git add .
git commit -m "Initial commit: Flower FL with FedAvg/FedProx/FedNova-like and plots"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
