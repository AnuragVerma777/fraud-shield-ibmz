# PROJECT CONTEXT — Fraud Shield on IBM Z

## Theme
**Real-Time AI for Critical Decisions** (IBM Datathon)

## Goal
Build a real-time payment fraud detection pipeline that:
- Scores each transaction in **milliseconds**
- Makes a three-way decision: **APPROVE / FLAG / BLOCK**
- Is designed to be deployable on **IBM Z** via **IBM Machine Learning for z/OS (Transactional AI)**

## Dataset
- **ULB Credit Card Fraud Detection** (Kaggle)
- Location: `data/creditcard.csv`
- 284,807 transactions, 31 columns (Time, V1–V28, Amount, Class)
- Highly imbalanced: ~0.17 % fraud (Class = 1)

## Tech Stack
| Layer | Tools |
|-------|-------|
| Data & ML | Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| IBM Z deployment | IBM ML for z/OS (ONNX export → scoring service) |

## Project Structure
```
IBM DATATHON/
├── data/                  # Raw & processed datasets
├── notebooks/             # Jupyter exploration notebooks
├── src/                   # Python source modules
│   ├── config.py          # Central config (paths, constants)
│   ├── data_loader.py     # Load & validate data
│   ├── feature_eng.py     # Feature engineering
│   ├── model.py           # Train, evaluate, export
│   ├── explainability.py  # SHAP explanations
│   └── scorer.py          # Real-time scoring simulator
├── app/                   # Streamlit dashboard
├── models/                # Saved model artifacts
├── docs/                  # Reports, charts, documentation
├── tests/                 # Unit & integration tests
├── .gitignore             # Ignore data/*.csv, models/, venv, __pycache__
├── requirements.txt       # Dependencies
├── PROJECT_CONTEXT.md     # THIS FILE — re-read every task
└── README.md              # Submission readme
```

## Coding Rules
1. Clean, commented, **beginner-readable** code
2. Small, single-purpose functions
3. **No hardcoded paths** — everything goes through `src/config.py`
4. After each task, document **how to run** and **expected output**

## Task Log
| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Project scaffold + config + data loader + EDA | ✅ Done | Structure, config, .gitignore, README, requirements, notebooks/01_eda.ipynb |
| 2 | Data pipeline (load, preprocess, split) | ✅ Done | src/data.py + tests/test_data.py; scaler → models/scaler.joblib |
| 3 | Model training (XGBoost + SMOTE) | ⬜ Pending | — |
| 4 | Evaluation & threshold tuning (APPROVE/FLAG/BLOCK) | ⬜ Pending | — |
| 5 | SHAP explainability | ⬜ Pending | — |
| 6 | Real-time scoring simulator | ✅ Done | src/stream.py + tests/test_stream.py; live log to data/stream_log.csv |
| 7 | Streamlit dashboard | ⬜ Pending | — |
| 8 | IBM Z deployment packaging (ONNX) | ⬜ Pending | — |
| 9 | Final README & presentation | ⬜ Pending | — |
