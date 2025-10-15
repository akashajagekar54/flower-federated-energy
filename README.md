# Flower Federated Learning: Energy Load Forecasting

This project demonstrates a **federated learning framework** for distributed energy load forecasting using [Flower](https://flower.dev) and PyTorch.  
Each client represents a building with its own dataset, and the server aggregates model updates using strategies like **FedAvg**, **FedProx**, and a **FedNova-like** normalization approach.

### ⚙️ Quickstart
```bash
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # (Windows)
# or
source .venv/bin/activate         # (Mac/Linux)

pip install -r requirements.txt

# Run Server
python -m src.flower_app.server --strategy fedavg --rounds 30

# Run Clients (in separate terminals)
python -m src.flower_app.client --cid 0 --server 127.0.0.1:8080
python -m src.flower_app.client --cid 1 --server 127.0.0.1:8080

# Plot results
python -m src.tools.plot_logs --server-logs runs/fedavg_logs/server_log.json
