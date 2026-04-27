# Model Card: E-commerce Purchase Intention

## Model Overview

This project trains baseline classifiers to predict whether an online shopping session will result in a purchase. The current MVP compares Logistic Regression and Random Forest models, then selects the best model by ROC-AUC on a deterministic test split.

## Intended Use

The model is intended for portfolio demonstration and local experimentation. It can support analysis of purchase-intent signals in session-level e-commerce data.

Appropriate uses include:

- learning how to structure a small MLOps project
- comparing baseline classifiers
- serving a local prediction API
- demonstrating reproducible model artifacts and tests

## Business Context

Lunera Commerce is a fictional European e-commerce company. The business question is whether visitor session behavior can help identify sessions with higher purchase likelihood.

Possible business applications include prioritizing remarketing analysis, studying funnel quality, or supporting experimentation. Any real intervention would require additional validation and business review.

## Dataset

The project uses the UCI Online Shoppers Purchasing Intention Dataset. The raw dataset is excluded from Git and should be placed at:

```text
data/raw/online_shoppers_intention.csv
```

The included sample CSV is demo data only and should not be used for model conclusions.

## Target

The target column is:

```text
Revenue
```

`Revenue` is converted into binary labels:

- `1`: session resulted in a purchase
- `0`: session did not result in a purchase

## Model Candidates

Current candidates:

- Logistic Regression with balanced class weights
- Random Forest Classifier with balanced class weights

Both are wrapped in scikit-learn pipelines that include preprocessing.

## Metrics

The training pipeline reports:

- ROC-AUC
- precision
- recall
- F1-score
- confusion matrix

ROC-AUC is used for model selection because the target can be imbalanced and threshold-independent ranking quality is useful at this stage.

## Limitations

- The model is trained on historical session-level data and may not generalize to different stores, traffic sources, regions, or seasons.
- Integer-coded categorical fields may hide more complex behavioral patterns.
- The MVP does not include probability calibration.
- The current evaluation uses a simple holdout split, not time-based validation.
- The repository does not include production monitoring, alerting, or automated retraining.

## Ethical And Business Considerations

Purchase-intent predictions should not be used to unfairly exclude users from offers, services, or support. If used in a real business setting, the team should review how predictions affect user experience and whether model behavior differs across relevant user segments.

The model should support decision-making, not replace human review for important business policy choices.

## Out-of-Scope Uses

This MVP should not be used for:

- credit, eligibility, or high-stakes user decisions
- direct personalization in production without validation
- real-time revenue reporting
- cross-company benchmarking
- legal or compliance conclusions

## Future Improvements

- Add calibration and threshold analysis.
- Compare models under time-aware validation.
- Add feature importance or explainability reporting.
- Track experiments after the MVP using a lightweight experiment tracker.
- Define retraining criteria based on drift and performance degradation.
