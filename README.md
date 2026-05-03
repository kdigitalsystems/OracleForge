# OracleForge

A dynamic ensemble trading engine utilizing a local LLM Mixture-of-Experts (MoE) architecture. 

OracleForge ingests daily market data and raw financial news to predict price action, employing closed-loop continuous learning and simulated take-profit/stop-loss execution to autonomously adjust model weights. 

State is managed entirely via JSON ledgers committed directly to this repository, ensuring the models' historical performance and current weights travel seamlessly with the codebase.

## Setup
1. `pip install -r requirements.txt`
2. Run `python update_tickers.py` to fetch the top 50 S&P companies.
3. Run `python forge_loop.py` daily after market close.
