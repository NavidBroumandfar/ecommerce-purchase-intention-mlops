# Changelog

## Unreleased

- Added documented baseline model results from the real UCI dataset.

## 0.1.0 - Portfolio MVP v1

Initial public portfolio version.

Included:

- local-first scikit-learn training pipeline
- preprocessing for mixed numeric, categorical, and boolean-like fields
- Logistic Regression and Random Forest baseline comparison
- ROC-AUC, precision, recall, F1-score, and confusion matrix reporting
- saved model, metrics, and metadata artifacts
- reusable prediction helper
- FastAPI `/health` and `/predict` endpoints
- pytest suite independent of the real UCI dataset
- Dockerfile and Makefile
- GitHub Actions CI
- model card and lightweight monitoring concept
- tracked sample dataset for tests and examples
