# Monitoring Concept

This project includes a lightweight local monitoring simulation. It is designed to show how monitoring would be approached, not to replace production observability.

## What Would Be Monitored

Key monitoring areas:

- input feature distributions
- prediction distribution
- model performance when labeled outcomes become available
- data quality issues such as missing values or unexpected categories
- service health for the FastAPI endpoint

## Feature Drift

Feature drift means the new prediction data no longer looks like the training or reference data. Examples include:

- higher `BounceRates`
- lower `PageValues`
- new or shifted `TrafficType` values
- seasonal changes in `Month`
- different proportions of `New_Visitor` and `Returning_Visitor`

The local drift script compares simple numeric summaries and categorical value shares between two CSV files.

## Prediction Distribution Drift

Prediction drift means the model starts producing a different mix of purchase probabilities or predicted classes. For example, a sudden increase in high purchase probabilities could indicate a real campaign effect, a data pipeline issue, or model instability.

In a production system, prediction distributions would be logged over time and compared against reference windows.

## Performance Degradation

True model performance can only be measured once actual purchase outcomes are available. The key metrics to monitor would be:

- ROC-AUC
- precision
- recall
- F1-score
- false positive and false negative counts

For an e-commerce use case, recall and precision should be reviewed in the context of business actions. A false positive may waste marketing budget, while a false negative may miss a valuable session.

## Retraining Trigger Concept

Retraining could be considered when:

- feature drift is persistent over multiple reporting windows
- prediction distribution changes sharply without a known business reason
- ROC-AUC or F1-score drops below an agreed threshold
- important categorical values appear that were not present during training
- the business changes the website, funnel, campaign strategy, or product catalog

The MVP does not retrain automatically. A human should review drift and performance reports before retraining.

## Limitations Of The Local Simulation

- It compares CSV files manually instead of scheduled production logs.
- It does not include alerting.
- It does not store historical monitoring windows.
- It does not connect predictions to delayed ground-truth labels.
- It does not replace model governance or incident response.

The goal is to demonstrate monitoring awareness while keeping the portfolio project simple and local-first.
