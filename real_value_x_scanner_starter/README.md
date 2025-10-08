# Real Value+ X Scanner — Quickstart

## 1) Install (Python 3.11 recommended)
pip install -r requirements.txt

## 2) Configure data sources
Edit `config/data_sources.yaml` and set your API keys / usernames.

## 3) Put your portfolio
Edit `data/portfolio.csv`.

## 4) Run (CLI examples)
python -m src.interfaces.cli portfolio enrich --in data/portfolio.csv --out out/portfolio_enriched.csv
python -m src.interfaces.cli scan --profile V1 --exclude-portfolio data/portfolio.csv --limit 50 --out out/v1_candidates.csv
python -m src.interfaces.cli analyze --ticker CEVA --model models/real_value_x_2025.yaml --out out/CEVA_report.md