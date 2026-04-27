# Portfolio Positioning

## One-Line GitHub Repo Description

End-to-end MLOps pipeline for predicting e-commerce purchase intention using the UCI Online Shoppers dataset.

## Suggested GitHub Topics

- machine-learning
- mlops
- scikit-learn
- fastapi
- docker
- github-actions
- classification
- ecommerce
- model-monitoring
- python
- portfolio-project

## GitHub Pinned-Repo Description Options

1. End-to-end ML/MLOps portfolio project for e-commerce purchase intention prediction with scikit-learn, FastAPI, Docker, CI, and model documentation.
2. Production-style MLOps pipeline for the UCI Online Shoppers dataset: training, evaluation, FastAPI serving, tests, Docker, and monitoring notes.
3. E-commerce purchase intention classifier with reproducible training, honest metrics, API serving, CI, Docker, and model lifecycle documentation.

## LinkedIn Project Blurb

I built an end-to-end MLOps portfolio project around e-commerce purchase intention prediction. The project uses the UCI Online Shoppers Purchasing Intention Dataset to model whether a website session ends in a purchase, then wraps the data science workflow in a local-first ML structure. It includes dataset validation, preprocessing for numeric, categorical, and boolean-like fields, baseline model comparison, artifact saving, model metadata, a generated results report, a model card, and a lightweight monitoring concept. The selected model is a Random Forest chosen by ROC-AUC, with real holdout results of ROC-AUC 0.916, precision 0.761, recall 0.474, and F1-score 0.584. The recall result is an important limitation: the model ranks sessions well, but the default threshold still misses many buyers. I also added FastAPI, Docker, pytest, and GitHub Actions CI to show how a standard classification use case can be made more reliable, reviewable, and MLOps-minded.

## LinkedIn Post Draft

I recently finished an MLOps portfolio project built around e-commerce purchase intention prediction.

The idea was to take a university-style data science use case and turn it into a production-oriented, local-first ML project without adding unnecessary infrastructure.

What it includes:

- UCI Online Shoppers dataset workflow
- reproducible preprocessing and train/test split
- Logistic Regression and Random Forest baselines
- model selection by ROC-AUC
- saved model, metrics, and metadata artifacts
- generated model results report and model card
- FastAPI `/health` and `/predict`
- Dockerfile, Makefile, pytest, and GitHub Actions CI
- lightweight monitoring concept for drift awareness

The best model was a Random Forest with ROC-AUC 0.916 on the holdout split. The limitation is just as important: recall is 0.474 at the default threshold, so this is not something I would call production-ready without threshold tuning, calibration, and business-cost analysis.

The main lesson was practical: good ML projects are not only about the model score. They are also about reproducibility, interfaces, testing, artifact hygiene, and communicating limitations clearly.

## How I Explain This Project In An Interview

### 30-Second Version

This is an end-to-end MLOps portfolio project for predicting whether an e-commerce session results in a purchase. I used the UCI Online Shoppers dataset, trained Logistic Regression and Random Forest baselines, selected Random Forest by ROC-AUC, and wrapped the workflow with tests, generated reports, a model card, FastAPI serving, Docker, and CI. The model has strong ROC-AUC at 0.916, but recall is only 0.474 at the default threshold, so I document it as a baseline that needs threshold tuning before real use.

### 90-Second Version

I wanted to show more than a notebook model, so I turned a standard e-commerce classification dataset into a small but complete ML project. The pipeline loads and validates the UCI Online Shoppers dataset, converts the `Revenue` target into binary labels, uses a deterministic stratified holdout split, and keeps preprocessing inside a scikit-learn pipeline to avoid leakage. Numeric features are scaled, and categorical or boolean-like fields are one-hot encoded.

For modeling, I compare Logistic Regression and Random Forest with balanced class weights and select by ROC-AUC because the target is imbalanced. The selected Random Forest achieved ROC-AUC 0.916, precision 0.761, recall 0.474, and F1-score 0.584. I call out the recall limitation because missed buyers matter in this business context, and the default threshold is not necessarily the right operating point.

On the MLOps side, the repo saves model, metric, and metadata artifacts locally, exposes predictions through FastAPI, includes a Dockerfile and Makefile, runs pytest in GitHub Actions, documents results in a model card, and includes a lightweight monitoring concept. It is intentionally scoped as a portfolio MVP rather than a full production system.

### Technical Deep-Dive Bullets

- Data handling: validates the expected UCI schema and keeps the raw dataset out of Git.
- Target handling: converts boolean or boolean-like `Revenue` values to binary labels.
- Split strategy: deterministic train/test split with stratification when class counts allow it.
- Leakage control: preprocessing is fit inside each scikit-learn pipeline using training data only.
- Feature handling: numeric fields are scaled; categorical and boolean-like fields are one-hot encoded.
- Model comparison: Logistic Regression and Random Forest are evaluated on the same holdout split.
- Selection logic: Random Forest is selected by ROC-AUC, not accuracy.
- Metrics: reports ROC-AUC, precision, recall, F1-score, and confusion matrix.
- API: FastAPI exposes `/health` without requiring a model and `/predict` with clear missing-artifact errors.
- Testing: pytest covers preprocessing, training, prediction, API behavior, and report generation without requiring real data or committed artifacts.
- MLOps scope: includes artifact hygiene, Docker, CI, model card, generated report, and monitoring notes without adding heavy infrastructure.
- Limitations: no production deployment, no MLflow registry, no live monitoring, no threshold tuning, and no probability calibration yet.

## Resume Bullet Options

- Built an end-to-end ML/MLOps portfolio project for e-commerce purchase intention prediction using scikit-learn, FastAPI, Docker, pytest, and GitHub Actions.
- Trained and compared Logistic Regression and Random Forest baselines on the UCI Online Shoppers dataset, selecting Random Forest by ROC-AUC with holdout ROC-AUC 0.916.
- Implemented a reproducible scikit-learn pipeline with target validation, stratified train/test splitting, numeric scaling, categorical encoding, artifact saving, and generated model reports.
- Developed a local FastAPI prediction service with `/health` and `/predict` endpoints, clear missing-artifact handling, Docker packaging, and CI-backed tests.
- Documented model results, limitations, monitoring considerations, and precision/recall trade-offs through a model card and public-facing project reports.
