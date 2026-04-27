# E-commerce Purchase Intention MLOps Pipeline

This repository turns a university-style classification dataset into a small, production-oriented machine learning project. It predicts whether an e-commerce website session is likely to result in a purchase.

The fictional business case is Lunera Commerce, a European online retailer that wants a practical way to identify high-intent visitor sessions and evaluate model behavior before investing in a larger deployment.

## Business Problem

E-commerce teams often need to decide which sessions deserve extra attention, such as personalized offers, remarketing, or sales funnel analysis. The goal here is to train a reproducible model that estimates purchase intent from session-level behavior.

The target variable is `Revenue`, which indicates whether the session ended in a purchase.

## Dataset

The project uses the UCI Online Shoppers Purchasing Intention Dataset.

The raw dataset is not committed to Git. Place it locally at:

```text
data/raw/online_shoppers_intention.csv
```

A small demo dataset is included at `data/sample/sample_online_shoppers.csv` for tests and API examples only.

## Repository Architecture

```text
.
├── data/
│   ├── README.md
│   └── sample/sample_online_shoppers.csv
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
├── artifacts/
├── Dockerfile
├── Makefile
└── .github/workflows/ci.yml
```

## Current MVP Scope

This is a local-first portfolio MVP. It includes:

- CSV loading and schema validation
- mixed-type preprocessing with scikit-learn
- reproducible train/test splitting
- Logistic Regression and Random Forest baselines
- ROC-AUC, precision, recall, F1-score, and confusion matrix reporting
- model and metadata artifacts saved with `joblib` and JSON
- reusable prediction helper
- FastAPI serving
- pytest coverage that does not require the raw dataset
- Docker support
- GitHub Actions CI
- a lightweight monitoring simulation concept

It intentionally does not include Kubernetes, Terraform, Airflow, cloud deployment, or MLflow.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
make install
```

Equivalent command:

```bash
python3 -m pip install -r requirements.txt
```

The Makefile defaults to `python3` for macOS/Linux compatibility. If your activated virtual environment exposes `python`, you can run commands such as `make test PYTHON=python`.

## Add the Dataset

Download the UCI Online Shoppers Purchasing Intention Dataset and place the CSV file here:

```text
data/raw/online_shoppers_intention.csv
```

The file must contain the original `Revenue` column. If the dataset is missing, training and evaluation fail with a clear message instead of a long traceback.

## Train the Model

```bash
make train
```

Equivalent command:

```bash
python3 -m src.models.train
```

Training saves:

```text
artifacts/model.joblib
artifacts/metrics.json
artifacts/model_metadata.json
```

These generated artifacts are excluded from Git.

## Evaluate

After training, run:

```bash
make evaluate
```

Equivalent command:

```bash
python3 -m src.models.evaluate
```

Evaluation uses the same deterministic split logic as training and reports ROC-AUC, precision, recall, F1-score, and the confusion matrix.

## Run a Local Prediction

After training, call the prediction helper from Python:

```python
from src.models.predict import predict_purchase

session = {
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
    "Weekend": False,
}

print(predict_purchase(session))
```

## Run the API

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

If `artifacts/model.joblib` is missing, the API returns a `503` response with instructions to train the model.

## Tests

```bash
make test
```

Equivalent command:

```bash
python3 -m pytest
```

The tests use synthetic and sample data, so they pass without the real UCI dataset.

## Docker

Build the image:

```bash
make docker-build
```

Run the API container:

```bash
make docker-run
```

Equivalent commands:

```bash
docker build -t ecommerce-purchase-intention-mlops .
docker run -p 8000:8000 ecommerce-purchase-intention-mlops
```

## CI/CD

GitHub Actions runs on every push and pull request. The workflow uses `actions/setup-python`, installs the Python dependencies, and runs `python -m pytest` in the CI environment. It does not require the raw dataset, which keeps CI reliable for public repository review.

## Metric Interpretation

- ROC-AUC measures ranking quality across thresholds and is the model selection metric.
- Precision answers: when the model predicts purchase, how often is it correct?
- Recall answers: of actual purchasing sessions, how many did the model catch?
- F1-score balances precision and recall.
- The confusion matrix shows true negatives, false positives, false negatives, and true positives.

Accuracy is not used as the main metric because online purchase datasets are often imbalanced.

## Monitoring Concept

The MVP includes a local monitoring simulation in `src/monitoring/drift_report.py`. It compares simple feature distributions between a reference CSV and a new CSV, then writes a JSON report under `reports/`.

Example:

```bash
python3 -m src.monitoring.drift_report \
  --reference data/sample/sample_online_shoppers.csv \
  --new data/sample/sample_online_shoppers.csv
```

This is not production monitoring. It is a lightweight demonstration of what would be tracked before a production system adds scheduled jobs, alerting, and model registry integration.

## Limitations

- The project is local-first and does not deploy to cloud infrastructure.
- The sample dataset is only for tests and examples.
- The baseline models are intentionally simple.
- The API expects a previously trained local model artifact.
- Monitoring is a simulation, not an operational alerting system.
- Business impact is not validated with live conversion or intervention data.

## Future Roadmap

- Add calibrated probability evaluation.
- Add threshold tuning based on business costs.
- Add richer drift checks and scheduled batch reports.
- Add experiment tracking, such as MLflow, after the MVP is stable.
- Add model version comparison documentation.
- Add a small front-end or dashboard only if it helps explain the workflow.
