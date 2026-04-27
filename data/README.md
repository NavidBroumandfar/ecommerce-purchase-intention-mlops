# Data

This project uses the UCI Online Shoppers Purchasing Intention Dataset.

The full raw dataset is intentionally excluded from Git because data files should not be committed directly to the portfolio repository.

## Expected Raw Dataset Location

Download the dataset from the UCI Machine Learning Repository and place the CSV file at:

```text
data/raw/online_shoppers_intention.csv
```

The training and evaluation scripts look for this path by default.

## Target Column

The target variable is:

```text
Revenue
```

`Revenue` indicates whether the visitor session resulted in a purchase. The preprocessing code converts boolean or string values such as `True` and `False` into binary labels.

## Sample Dataset

A small tracked sample is included at:

```text
data/sample/sample_online_shoppers.csv
```

This sample is demo data only. It follows the expected UCI-style schema and is used for tests, examples, and API payloads. It is not intended for real model training or business conclusions.
