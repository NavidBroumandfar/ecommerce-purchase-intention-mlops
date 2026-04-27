# Model Card: E-commerce Purchase Intention

## Model Purpose

This model estimates whether an e-commerce visitor session is likely to result in a purchase. It is part of a portfolio project with production-style structure that demonstrates a local-first MLOps workflow around a standard classification problem.

The model is not presented as a deployed production system. It is intended to show clear data handling, preprocessing, training, evaluation, artifact management, API serving, and lifecycle documentation.

## Intended Users

Intended users for this repository are:

- recruiters and hiring managers reviewing an ML/MLOps portfolio project
- data science practitioners studying a compact project structure
- developers who want a local FastAPI example around a trained scikit-learn pipeline

## Intended Use

Appropriate uses include:

- demonstrating a reproducible ML training workflow
- comparing simple baseline classifiers
- generating local predictions for example visitor sessions
- reviewing model metrics and limitations
- discussing how monitoring and retraining could be approached

In a real business context, similar predictions could support funnel analysis, remarketing prioritization, or experimentation planning. Those uses would require additional validation before any customer-facing action.

## Out-of-Scope Use

This MVP should not be used for:

- high-stakes decisions about individuals
- eligibility, credit, insurance, employment, or legal decisions
- direct production personalization without validation
- automated discounting or exclusion of users from offers
- real-time revenue reporting
- cross-company performance benchmarking

## Business Context

Lunera Commerce is a fictional European e-commerce company. The business question is whether session-level browsing behavior can help identify purchase intent before or during a website visit.

The business value would come from understanding behavioral patterns and improving decision support. The model should be treated as one input into analysis, not as a replacement for product, marketing, or customer experience review.

## Dataset

Dataset: UCI Online Shoppers Purchasing Intention Dataset

The raw dataset is excluded from Git and should be placed at:

```text
data/raw/online_shoppers_intention.csv
```

The dataset contains session-level fields such as page counts, page durations, bounce and exit rates, page values, month, visitor type, traffic type, browser, operating system, region, weekend indicator, and purchase outcome.

The repository includes a small sample CSV for tests and examples only. The sample data is not suitable for model evaluation or business conclusions.

## Target Variable

Target column:

```text
Revenue
```

The target is converted to binary labels:

- `1`: session resulted in a purchase
- `0`: session did not result in a purchase

## Model Candidates

Current candidates:

- Logistic Regression with balanced class weights
- Random Forest Classifier with balanced class weights

Both candidates are trained as scikit-learn pipelines that combine preprocessing and the estimator. The preprocessing step scales numeric features and one-hot encodes categorical or boolean-like features.

## Metrics

The training pipeline reports:

- ROC-AUC
- precision
- recall
- F1-score
- confusion matrix

ROC-AUC is used for model selection because purchase outcomes are commonly imbalanced and ranking quality is useful before choosing an operating threshold.

## Current Baseline Results

Evaluation date: `2026-04-27T17:52:28.949481+00:00`

Dataset used: local copy of the real UCI dataset at `data/raw/online_shoppers_intention.csv`

Dataset shape: `12,330` sessions x `18` columns

Target distribution:

| Revenue | Sessions | Share |
| --- | ---: | ---: |
| False | 10,422 | 84.53% |
| True | 1,908 | 15.47% |

Model comparison:

| Model | ROC-AUC | Precision | Recall | F1-score | Confusion Matrix [[TN, FP], [FN, TP]] |
| --- | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.893 | 0.491 | 0.743 | 0.592 | [[1790, 294], [98, 284]] |
| **Random Forest** | 0.916 | 0.761 | 0.474 | 0.584 | [[2027, 57], [201, 181]] |

Selected model: `random_forest`

Selection metric: `roc_auc`

The selected model's ROC-AUC is strong for a baseline holdout result, but recall is moderate/low at the default threshold: 201 purchasing sessions were false negatives and 57 non-purchasing sessions were false positives. In business terms, false negatives are missed buyers and false positives can drive unnecessary targeting or intervention.

These are baseline holdout results for the portfolio MVP. They should not be interpreted as production performance guarantees.

## Limitations

- The current evaluation uses a deterministic holdout split, not time-based validation.
- The model may not generalize to different stores, geographies, traffic sources, product catalogs, or seasonal periods.
- Some categorical fields are integer-coded, which can hide underlying category meaning.
- Probability calibration is not included in the MVP.
- Threshold selection is not tied to an explicit business cost matrix.
- The project does not include live monitoring, alerting, or automated retraining.

## Potential Bias And Business Risks

The dataset describes browsing sessions rather than protected characteristics, but business risk can still appear through proxy patterns such as geography, device, traffic source, or visitor type.

Potential risks include:

- over-targeting certain user segments with marketing interventions
- missing high-intent sessions from underrepresented traffic patterns
- reacting to seasonal drift as if it were a stable signal
- using model scores without reviewing false positives and false negatives
- applying the model to a store or market that differs from the training data

Any real use should include segment-level review, business-owner signoff, and monitoring after deployment.

## Monitoring Needs

A practical monitoring plan would track:

- input feature drift
- missing values and unexpected categories
- prediction probability distribution
- predicted class distribution
- delayed ground-truth performance when purchase outcomes are available
- service health for the API

The current repository includes a local drift-report simulation only. It does not include production logging, scheduled jobs, alerting, or model registry integration.

## Future Improvements

- Add probability calibration analysis.
- Add threshold tuning based on business costs.
- Compare validation strategies, especially time-aware splits if timestamps are available.
- Add feature importance or explainability reporting.
- Track experiment runs after the MVP is stable.
- Define formal retraining criteria based on drift and performance degradation.
