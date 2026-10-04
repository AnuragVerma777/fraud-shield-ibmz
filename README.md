# Fraud Shield on IBM Z

**IBM Datathon — Real-Time AI for Critical Decisions**

Fraud Shield explores a real-time payment fraud detection pipeline. The goal
is to score transactions quickly and support three decisions: **APPROVE**,
**FLAG for review**, or **BLOCK**. The project is designed with a potential
deployment on IBM Z using IBM Machine Learning for z/OS in mind.

## Problem statement

Payment fraud detection is an imbalanced classification problem: fraudulent
transactions are rare, but failing to detect them can be costly. A useful
system must identify fraud while limiting false alarms that disrupt genuine
customers. Day 1 establishes a reproducible data split, compares two
class-weighted classifiers, and examines how the probability threshold
changes precision and recall. Thresholds are a starting point for the
APPROVE / FLAG / BLOCK policy, not a substitute for business validation.

## Dataset

This project uses the [ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
It contains 284,807 European card transactions from September 2013, with 492
fraud cases. The 31 columns are `Time`, `Amount`, 28 anonymized PCA features
(`V1`–`V28`), and the target `Class` (`1` = fraud, `0` = legitimate). Fraud
accounts for about 0.17% of rows, so accuracy alone can hide poor fraud
detection.

Download `creditcard.csv` from Kaggle and place it at `data/creditcard.csv`.
The data pipeline makes a stratified 60/20/20 train/validation/test split and
fits the `Time` and `Amount` scaler on the training data only, then applies it
to the other splits.

## Day 1 setup and run

Use Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the downloaded CSV at `data/creditcard.csv`, then train and evaluate
the models:

```powershell
python -m src.train
```

This compares Logistic Regression (`class_weight="balanced"`) and XGBoost
(with `scale_pos_weight` calculated from the training labels). It prints
precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrices, then saves:

- `models/xgb_model.joblib` — trained XGBoost model
- `models/scaler.joblib` — preprocessing scaler
- `docs/metrics_day1.json` — metrics for both models
- `docs/pr_curve.png` — combined precision-recall plot
- `docs/confusion_matrix.png` — XGBoost confusion matrix

Inspect threshold trade-offs using the saved XGBoost model:

```powershell
python -m src.threshold
```

This reports precision and recall across thresholds from 0.1 to 0.9, prints
candidate BLOCK and FLAG thresholds with false-positive/false-negative
counts, and saves `docs/threshold_tradeoff.png`. Run training first. The
threshold suggestions are measured on the test set for exploration; choose
production thresholds using a separate validation process and business costs.

## Day 1 results

The results below are measured on the held-out test set at a probability
threshold of 0.5. The test set contains 56,962 transactions. Full metrics and
confusion matrices are in `docs/metrics_day1.json`.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0613 | 0.9184 | 0.1149 | 0.9726 | 0.7209 |
| XGBoost | 0.8817 | 0.8367 | 0.8586 | 0.9762 | 0.8796 |

## Day 2: combined risk evaluation

The combined risk score is calculated on a 0–100 scale:

```text
risk = 100 × (w_xgb × XGBoost fraud probability
              + w_anom × normalized Isolation Forest anomaly score)
```

Weights and decision thresholds were selected using the validation set. The
best PR-AUC among the tested weight options was obtained with **XGBoost 1.0**
and **anomaly 0.0**. The saved thresholds in `models/risk_config.json` are:

- **FLAG:** 0.0000094
- **BLOCK:** 46.6832

The FLAG cutoff is extremely low: on the test set it sends 56,850 of 56,864
genuine transactions to review. This follows the tuning rule of choosing the
lowest score threshold that meets the recall target, but creates an impractical
review workload. Treat these as experimental results, not production settings;
a usable policy also needs a review-capacity or minimum-precision constraint.

### Test-set ranking results

| Scoring method | PR-AUC | ROC-AUC |
|---|---:|---:|
| XGBoost alone | 0.8796 | 0.9762 |
| Combined risk score | 0.8796 | 0.9762 |

The combined score did **not** outperform XGBoost on either ranking metric.
Because the selected anomaly weight is zero, these scores are identical. An
anomaly detector can still provide a complementary signal by surfacing
transactions that differ from known legitimate behavior, including possible
new fraud patterns. This test does not demonstrate that it caught previously
unseen fraud; the validation results simply did not support giving it weight
in the current combined score.

### Three-way test decisions

| Actual class | APPROVE | FLAG | BLOCK | Total |
|---|---:|---:|---:|---:|
| Genuine | 3 | 56,850 | 11 | 56,864 |
| Fraud | 0 | 16 | 82 | 98 |

The policy caught 98 fraud cases through BLOCK or FLAG and approved none of
the fraud cases. It incorrectly blocked 11 genuine customers and flagged
56,850 genuine transactions. Scoring latency was **16.142 ms average** and
**23.697 ms at p95**, measured from 1,000 individual RiskEngine calls sampled
across the test set. Detailed numbers are in `docs/metrics_day2.json`;
charts are `docs/day2_model_comparison.png` and
`docs/day2_decision_matrix.png`.

## Project structure

```text
├── data/          Dataset files (download separately)
├── docs/          Metrics and generated plots
├── models/        Saved model and scaler artifacts
├── notebooks/     Exploration notebooks
├── src/           Data preparation, training, and threshold analysis
└── tests/         Data pipeline tests
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · joblib · Matplotlib

## License

MIT
