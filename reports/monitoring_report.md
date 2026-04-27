# Monitoring Concept

This project includes a lightweight local monitoring simulation. It is meant to demonstrate monitoring awareness in a portfolio MVP, not to replace production observability.

The current script compares two CSV files and writes a JSON summary of basic distribution differences. It is useful for explaining the concept of drift, but it does not include scheduled jobs, alerting, dashboards, historical storage, or production incident response.

## What Data Drift Means

Data drift occurs when new input data no longer looks like the data used to train or validate the model.

For this e-commerce use case, drift could appear as:

- higher or lower `BounceRates`
- changes in `PageValues`
- shifts in `TrafficType`
- seasonal changes in `Month`
- a different mix of `New_Visitor`, `Returning_Visitor`, and `Other`
- new or unexpected category values
- changes in missing-value rates

The local drift script compares numeric summaries and categorical value shares between a reference dataset and a new dataset.

## What Prediction Drift Means

Prediction drift occurs when the model starts producing a different distribution of outputs.

Examples:

- the average purchase probability changes sharply
- the share of sessions predicted as likely purchases increases or decreases
- predictions cluster near 0 or 1 more than expected
- the model produces high-confidence predictions for traffic that looks unfamiliar

Prediction drift is not automatically bad. It may reflect a real business event, such as a campaign or seasonal sale. It should trigger investigation rather than an automatic conclusion.

## Performance Monitoring

Performance monitoring requires ground-truth purchase outcomes after predictions are made. Without delayed labels, the team can monitor inputs and predictions, but it cannot confirm whether the model is still accurate.

When labels become available, useful metrics include:

- ROC-AUC
- precision
- recall
- F1-score
- false positives and false negatives
- segment-level performance by traffic source, visitor type, region, or device category

For a business workflow, these metrics should be reviewed alongside the cost of wrong actions. A false positive may waste marketing budget, while a false negative may miss a valuable session.

## Retraining Trigger Concept

A retraining review could be triggered when:

- important features drift for multiple reporting windows
- prediction distributions shift without a known business reason
- labeled performance falls below an agreed threshold
- new categories appear frequently
- the website, product catalog, marketing strategy, or traffic mix changes
- business owners change the objective or intervention tied to the model

The MVP does not retrain automatically. A human should review monitoring evidence before retraining or replacing a model.

## Why This Is Not Enterprise Monitoring

This repository intentionally stays local-first. The monitoring component does not include:

- production data logging
- streaming metrics
- scheduled batch jobs
- alert routing
- dashboards
- model registry integration
- audit trails
- automated retraining

Those additions would be reasonable future work for a larger system, but they are outside the scope of this portfolio MVP.

## Local Script

Generate a simple drift-style JSON report:

```bash
python3 -m src.monitoring.drift_report \
  --reference data/sample/sample_online_shoppers.csv \
  --new data/sample/sample_online_shoppers.csv
```

Default output:

```text
reports/drift_report.json
```

The generated JSON report is ignored by Git.
