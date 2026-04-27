import pandas as pd

from src.features.preprocessing import build_preprocessor, prepare_features_and_target, split_features_target


def _synthetic_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Administrative": [0, 1, 2, 3],
            "Administrative_Duration": [0.0, 12.5, 30.0, 45.5],
            "Month": ["Feb", "Mar", "May", "Nov"],
            "VisitorType": ["Returning_Visitor", "New_Visitor", "Returning_Visitor", "Other"],
            "Weekend": [False, True, False, True],
            "Revenue": ["False", "True", "False", "True"],
        }
    )


def test_split_features_target_returns_binary_target() -> None:
    df = _synthetic_dataframe()

    X, y = split_features_target(df)

    assert "Revenue" not in X.columns
    assert len(X) == len(y)
    assert set(y.tolist()) == {0, 1}


def test_preprocessing_pipeline_fit_transforms_mixed_features() -> None:
    df = _synthetic_dataframe()
    X, y, preprocessor = prepare_features_and_target(df)

    transformed = preprocessor.fit_transform(X, y)

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= 1


def test_build_preprocessor_handles_boolean_categorical_columns() -> None:
    df = _synthetic_dataframe()
    X, _ = split_features_target(df)

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(df)
