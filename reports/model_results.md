# Model Results

## Evaluation Summary

Evaluation date: `2026-04-27T17:52:28.949481+00:00`

Dataset source: UCI Online Shoppers Purchasing Intention Dataset

Official dataset page: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

Official download URL used locally: https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip

Local dataset path: `data/raw/online_shoppers_intention.csv`

Target variable: `Revenue`

Dataset shape: `12,330` rows x `18` columns

## Target Distribution

| Revenue | Sessions | Share |
| --- | ---: | ---: |
| False | 10,422 | 84.53% |
| True | 1,908 | 15.47% |

The dataset is imbalanced: most sessions do not end in purchase. This makes precision and recall more informative than accuracy alone.

## Train/Test Split

- Split approach: deterministic holdout split
- Test size: `0.2`
- Stratified split: `True`
- Random state: `42`
- Train rows: `9,864`
- Test rows: `2,466`

## Model Candidates

- Logistic Regression with balanced class weights
- Random Forest Classifier with balanced class weights

Both candidates use the same scikit-learn preprocessing pipeline: numeric features are scaled and categorical or boolean-like features are one-hot encoded.

## Model Comparison

| Model | ROC-AUC | Precision | Recall | F1-score | Confusion Matrix [[TN, FP], [FN, TP]] |
| --- | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.893 | 0.491 | 0.743 | 0.592 | [[1790, 294], [98, 284]] |
| **Random Forest** | 0.916 | 0.761 | 0.474 | 0.584 | [[2027, 57], [201, 181]] |

## Selected Model

Selected best model: `random_forest`

Selection metric: `roc_auc`

Best model metrics:

- ROC-AUC: `0.916`
- Precision: `0.761`
- Recall: `0.474`
- F1-score: `0.584`
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[2027, 57], [201, 181]]`

## Business Interpretation

The Random Forest baseline achieved the strongest ROC-AUC in this run, which means it ranked purchasing sessions better than the other baseline candidate on the deterministic test split. Its ROC-AUC of `0.916` is a strong baseline ranking result on this holdout split.

The default classification threshold is more conservative: recall is `0.474`, with `201` false negatives and `57` false positives on the test split. False negatives are missed buyers; false positives are non-purchasing sessions that could receive unnecessary targeting or intervention.

This is why the precision/recall trade-off matters for this use case. A business team would need to choose an operating threshold based on the cost of missed purchases versus unnecessary interventions.

## Limitations

- Results are based on the local UCI CSV at `data/raw/online_shoppers_intention.csv`.
- The raw dataset is not committed to Git.
- The evaluation uses a deterministic holdout split, not time-based validation.
- The model is a baseline portfolio model, not a production-ready decision system.
- Probability calibration and threshold tuning are not included in this MVP.
- Business impact is not validated with live experiments or intervention outcomes.
