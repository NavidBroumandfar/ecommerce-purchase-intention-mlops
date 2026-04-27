# E-commerce Purchase Intention MLOps Pipeline

[![CI](https://github.com/NavidBroumandfar/ecommerce-purchase-intention-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/NavidBroumandfar/ecommerce-purchase-intention-mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Docker](https://img.shields.io/badge/Docker-supported-2496ED)

This repository is a production-oriented portfolio project that predicts whether an e-commerce website session is likely to end in a purchase.

It uses the UCI Online Shoppers Purchasing Intention Dataset and wraps a standard classification task in a local-first MLOps workflow: reproducible preprocessing, model training, evaluation, artifact saving, prediction serving, tests, Docker, CI, and lightweight monitoring documentation.

The project is intentionally scoped as a clean portfolio MVP. It is designed to demonstrate reproducibility, testing, API serving, and model lifecycle thinking without adding unnecessary infrastructure.

## Overview

The project trains baseline machine learning models on session-level browsing behavior, selects the best candidate by ROC-AUC, saves the trained pipeline as a local artifact, and exposes predictions through a FastAPI endpoint.

Core workflow:

1. Load and validate the dataset.
2. Split features from the `Revenue` target.
3. Build a preprocessing pipeline for numeric, categorical, and boolean-like fields.
4. Train Logistic Regression and Random Forest candidates.
5. Evaluate with ROC-AUC, precision, recall, F1-score, and confusion matrix.
6. Save the best model and metadata under `artifacts/`.
7. Serve predictions through a local FastAPI app.

## Business Problem

Lunera Commerce is a fictional European e-commerce company that wants to understand which visitor sessions show purchase intent.

Purchase intention prediction matters because many commercial actions depend on identifying high-intent or low-intent sessions early enough to act. Examples include funnel diagnostics, campaign analysis, remarketing prioritization, or experimentation around offers and checkout flows.

This repository does not claim to automate those decisions in production. It shows how a university-style data science use case can be structured as a maintainable ML project with clear interfaces, tests, artifacts, and operational thinking.

## Dataset

Dataset: UCI Online Shoppers Purchasing Intention Dataset

Target variable: `Revenue`

`Revenue` indicates whether a visitor session resulted in a purchase. The preprocessing code converts boolean or boolean-like string values into binary labels:

- `1`: purchase
- `0`: no purchase

The real dataset is not committed to Git. Place it locally at:

```text
data/raw/online_shoppers_intention.csv
```

A small tracked sample is included at:

```text
data/sample/sample_online_shoppers.csv
```

The sample data is for tests, examples, and schema demonstration only. It should not be used for model conclusions.

## Architecture

The project keeps the workflow simple and inspectable:

- `src/data`: CSV loading and dataset validation
- `src/features`: feature/target splitting and scikit-learn preprocessing
- `src/models`: training, evaluation, artifact loading, and prediction helpers
- `src/api`: FastAPI serving layer
- `src/monitoring`: local drift-report simulation
- `tests`: fast pytest coverage using synthetic/sample data
- `reports`: model card and monitoring concept notes
- `artifacts`: generated model and metric outputs, excluded from Git except `.gitkeep`

## Project Structure

```text
.
├── README.md
├── CHANGELOG.md
├── requirements.txt
├── Dockerfile
├── Makefile
├── .github/workflows/ci.yml
├── data/
│   ├── README.md
│   └── sample/sample_online_shoppers.csv
├── notebooks/
│   └── .gitkeep
├── src/
│   ├── data/load_data.py
│   ├── features/preprocessing.py
│   ├── models/train.py
│   ├── models/evaluate.py
│   ├── models/predict.py
│   ├── api/main.py
│   └── monitoring/drift_report.py
├── tests/
├── reports/
│   ├── model_card.md
│   └── monitoring_report.md
└── artifacts/
    └── .gitkeep
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

Equivalent install command:

```bash
python3 -m pip install -r requirements.txt
```

The Makefile defaults to `python3` for macOS/Linux compatibility:

```makefile
PYTHON ?= python3
```

If your activated virtual environment exposes `python` and you prefer that command, use:

```bash
make test PYTHON=python
```

## Training

First place the real UCI CSV at:

```text
data/raw/online_shoppers_intention.csv
```

Then run:

```bash
make train
```

Equivalent command:

```bash
python3 -m src.models.train
```

Training writes generated artifacts to:

```text
artifacts/model.joblib
artifacts/metrics.json
artifacts/model_metadata.json
```

These files are intentionally ignored by Git. If the raw dataset is missing, the command exits with a clear message explaining where to place it.

## Evaluation

After training, evaluate the saved model:

```bash
make evaluate
```

Equivalent command:

```bash
python3 -m src.models.evaluate
```

The evaluation script loads `artifacts/model.joblib`, recreates the deterministic test split, and prints ROC-AUC, precision, recall, F1-score, and the confusion matrix.

## API Usage

Start the local API:

```bash
make api
```

Equivalent command:

```bash
python3 -m uvicorn src.api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "project": "ecommerce-purchase-intention-mlops"
}
```

Example prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Administrative": 1,
    "Administrative_Duration": 18.2,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 12,
    "ProductRelated_Duration": 340.1,
    "BounceRates": 0.01,
    "ExitRates": 0.03,
    "PageValues": 12.3,
    "SpecialDay": 0.0,
    "Month": "Mar",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 3,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": false
  }'
```

Successful prediction responses include this structure. The class, probability, and selected model name depend on the trained artifact:

```json
{
  "predicted_class": 1,
  "purchase_probability": 0.82,
  "model_version": "portfolio-mvp-v1",
  "model_metadata": {
    "best_model_name": "random_forest",
    "target_column": "Revenue"
  }
}
```

`model_metadata` is populated from `artifacts/model_metadata.json` when that file exists. The real metadata contains additional training details. If the model artifact is missing, `/predict` returns HTTP `503` with a clear message. `/health` works whether or not a model has been trained.

## Docker Usage

Build the API image:

```bash
make docker-build
```

Run the container:

```bash
make docker-run
```

Equivalent commands:

```bash
docker build -t ecommerce-purchase-intention-mlops .
docker run -p 8000:8000 ecommerce-purchase-intention-mlops
```

The container starts the FastAPI application on port `8000`. As with local API usage, predictions require a trained model artifact to be available inside the container.

## Testing

Run the test suite:

```bash
make test
```

Equivalent command:

```bash
python3 -m pytest
```

The tests are fast and do not require the real UCI dataset or committed model artifacts. They cover preprocessing, training on synthetic data, prediction with a temporary model artifact, and API behavior.

## CI/CD

GitHub Actions runs on push and pull request. The workflow:

1. checks out the repository
2. sets up Python 3.11
3. installs dependencies
4. runs `python -m pytest`

CI intentionally does not depend on `data/raw/` or generated model artifacts.

## MLOps Components

Included in this MVP:

- reproducible Python package structure
- data validation for the expected target column
- scikit-learn preprocessing and model pipelines
- deterministic train/test split
- baseline model comparison
- metric and metadata artifact saving
- reusable prediction helper
- FastAPI serving
- pytest coverage
- Dockerfile for local API packaging
- GitHub Actions CI
- model card
- lightweight monitoring simulation

Not included by design:

- cloud deployment
- Kubernetes
- Terraform
- Airflow
- MLflow or model registry integration
- production alerting or automated retraining

## Model Card

The model card is available at:

```text
reports/model_card.md
```

It documents the model purpose, intended use, model candidates, evaluation metrics, limitations, business risks, monitoring needs, and future improvements.

## Monitoring Concept

The monitoring note is available at:

```text
reports/monitoring_report.md
```

The local monitoring script can compare simple feature distributions between two CSV files:

```bash
python3 -m src.monitoring.drift_report \
  --reference data/sample/sample_online_shoppers.csv \
  --new data/sample/sample_online_shoppers.csv
```

This writes a JSON report to `reports/drift_report.json`, which is ignored by Git.

## Limitations

- This is a local-first portfolio MVP, not a full production system.
- The real UCI dataset is excluded from Git and must be added locally.
- The included sample CSV is only for tests and examples.
- The current validation uses a deterministic holdout split, not time-based validation.
- The baseline models are intentionally simple.
- Probability calibration and threshold tuning are not included yet.
- Monitoring is a local simulation and does not include alerts, scheduled jobs, or historical tracking.
- Business impact is not validated with live experiments or intervention outcomes.

## Roadmap

Practical next steps:

- add threshold tuning based on business costs
- add probability calibration analysis
- compare models with time-aware validation if timestamped data is available
- add a lightweight experiment tracking option after the MVP is stable
- extend monitoring reports with prediction distribution checks
- document retraining criteria more formally
