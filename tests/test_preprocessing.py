import numpy as np
import pandas as pd
import pytest
from src.config import MMINS_CAP, RANDOM_STATE, TARGET
from src.dataset_cleaning_and_preprocessing import (
    add_derived_features_and_clean,
    create_eda_preprocessor,
    create_imputed_dataframe,
    create_imputer_preprocessor,
    filter_columns_from_training_data,
    get_feature_groups,
    get_missing_percentages_per_column,
    get_missing_percentages_per_row,
    ordinal_features_mapping,
    prepare_data_for_splitting,
    remove_invalid_rows,
    split_data,
)
pd.set_option('future.no_silent_downcasting', True) #To avoid warnings with test_ordinal_features_mapping_uses_expected_codes()

def create_minimal_raw_dataframe() -> pd.DataFrame:

    return pd.DataFrame(

        {
            TARGET: [500.0, 550.0, 600.0, 450.0],
            "CNTSCHID": [800063, 800071, 800080, 800080],
            "CNTSTUID": [800001, 800002, 800283, 800431],
            "ESCS": [0.0, 1.0, -1.0, 1.0],
            "ST059Q01TA": [5.0, 4.0, 10.0, 2.0],
            "SC175Q01JA": [45.0, 50.0, 55.0, 30.0],
            "REPEAT": [
                "Never repeated",
                "Repeated at least once",
                "Never repeated",
                "Never repeated",
            ],

            "ST062Q01TA": [
                "Never",
                "One or two times",
                "Three or four times",
                "Five or more times",
            ],

            "IMMIG": [
                "Native student",
                "Second-Generation student",
                "First-Generation student",
                "Native student",
            ],

            "ST004D01T": [
                "Female",
                "Male",
                "Female",
                "Male",
            ],

            "SCHLTYPE": [
                "Private independent",
                "Private Government-dependent",
                "Public",
                "Public",
            ],

            "MATHEFF": [-2.0, 0.5, 1.0, 2.0],
        }
    )

def test_add_derived_features_creates_mmins():

    df = create_minimal_raw_dataframe()
    result = add_derived_features_and_clean(df)
    expected_mmins = pd.Series([225.0, 200.0, 550.0, 60.0], name="MMINS")
    expected_mmins.value_counts(sort=False).eq(result.value_counts(sort=False)).all() #REFERENCE: https://stackoverflow.com/q/75133025, Posted by the phoenix, Retrieved 2026-07-28, License - CC BY-SA 4.0

def test_add_derived_features_creates_school_mean_escs():

    df = create_minimal_raw_dataframe()
    result = add_derived_features_and_clean(df)
    expected = pd.Series([0.5, 0.5, 0.0, 0.0], name="MEAN_ESCS")
    expected.value_counts(sort=False).eq(result.value_counts(sort=False)).all()

def test_add_derived_features_removes_source_and_identifier_columns():

    df = create_minimal_raw_dataframe()
    result = add_derived_features_and_clean(df)
    removed_columns = {"ESCS", "ST059Q01TA", "SC175Q01JA", "CNTSCHID", "CNTSTUID"}

    assert removed_columns.isdisjoint(result.columns) # REFERENCE: https://www.geeksforgeeks.org/python/python-set-isdisjoint-method/ 
    assert "MMINS" in result.columns
    assert "MEAN_ESCS" in result.columns

def test_add_derived_features_does_not_modify_input_dataframe():

    df = create_minimal_raw_dataframe()
    original = df.copy(deep=True)
    add_derived_features_and_clean(df)
    pd.testing.assert_frame_equal(df, original) # "assert_frame_equal" is preferred over "equal" because it throws an AssertionError when two DataFrames aren't equal (REFERENCE: https://stackoverflow.com/questions/51017249/what-is-the-difference-between-assert-frame-equal-and-equals)

def test_ordinal_features_mapping_uses_expected_codes():

    df = pd.DataFrame({ "REPEAT": ["Never repeated", "Repeated at least once"], "ST062Q01TA": ["Never", "Five or more times"], "IMMIG": ["Native student", "First-Generation student"]})
    result = ordinal_features_mapping(df)
    
    assert result["REPEAT"].tolist() == [0, 1]
    assert result["ST062Q01TA"].tolist() == [1, 4]
    assert result["IMMIG"].tolist() == [1, 3]


def test_ordinal_features_mapping_does_not_modify_input():

    df = pd.DataFrame({"REPEAT": ["Never repeated"], "ST062Q01TA": ["Never"], "IMMIG": ["Native student"]})
    original = df.copy(deep=True)
    ordinal_features_mapping(df)
    pd.testing.assert_frame_equal(df, original)

def test_missing_percentages_per_column():

    df = pd.DataFrame({"A": [1.0, np.nan, 3.0, np.nan], "B": [np.nan, 2.0, 3.0, 4.0], "C": [1.0, 2.0, 3.0, 4.0]})
    result = get_missing_percentages_per_column(df)
    assert result["A"] == pytest.approx(50.0)
    assert result["B"] == pytest.approx(25.0)
    assert result["C"] == pytest.approx(0.0)

def test_remove_invalid_rows_with_missing_target():

    df = pd.DataFrame({TARGET: [500.0, np.nan, 600.0, np.nan], "A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 5.0, 6.0, np.nan]})

    result = remove_invalid_rows(df, threshold=70.0)

    assert len(result) == 2
    assert result[TARGET].notna().all()

def test_remove_invalid_rows_removes_excessive_feature_missingness():

    df = pd.DataFrame({TARGET: [500.0, 550.0, 600.0], "A": [1.0, np.nan, np.nan], "B": [2.0, np.nan, np.nan], "C": [3.0, 4.0, np.nan], "D": [4.0, 5.0, 6.0]})
    result = remove_invalid_rows(df, threshold = 50.0)

    # Row 0: 0% missing -> retained
    # Row 1: 50% missing -> removed because condition is < 50 (TARGET is excluded)
    # Row 2: 75% missing -> removed

    assert result.index.tolist() == [0]


#REFERENCE: https://docs.pytest.org/en/7.1.x/how-to/assert.html 
def test_remove_invalid_rows_raises_error_when_target_is_missing():
    
    df = pd.DataFrame({"A": [1.0, 2.0]})
    with pytest.raises(ValueError, match = "Target variable"):
        remove_invalid_rows(df, threshold=70.0)


def test_filter_columns_uses_training_missingness_only():

    X_train = pd.DataFrame({"ST059Q01TA": [1.0, 2.0, 3.0, 4.0], "SC175Q01JA": [np.nan, np.nan, np.nan, 45]})

    # The test set deliberately has no missing value in "SC175Q01JA".
    # This must not affect the training selection.

    X_test = pd.DataFrame({"ST059Q01TA": [5.0, 6.0], "SC175Q01JA": [45.0, 60.0]})

    (X_train_filtered, X_test_filtered, removed_columns) = filter_columns_from_training_data(X_train, X_test, threshold=70.0,)

    assert X_train_filtered.columns.tolist() == ["ST059Q01TA"]
    assert X_test_filtered.columns.tolist() == ["ST059Q01TA"]
    assert removed_columns == ["SC175Q01JA"]


def test_filter_columns_removes_column_exactly_at_threshold():

    X_train = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [np.nan, np.nan, 3.0, 4.0]})
    X_test = pd.DataFrame({"A": [5.0], "B": [6.0]})

    (X_train_filtered, X_test_filtered, removed_columns) = filter_columns_from_training_data(X_train, X_test, threshold = 50.0)

    assert "B" not in X_train_filtered.columns
    assert "B" not in X_test_filtered.columns
    assert removed_columns == ["B"]

def test_split_data_with_correct_test_size():

    df = pd.DataFrame({TARGET: np.random.uniform(200, 500, size = 20), "A": np.arange(20, dtype=float)})

    X_train, X_test, y_train, y_test = split_data(df, test_size = 0.25, random_state=RANDOM_STATE)

    assert len(X_train) == 15
    assert len(X_test) == 5
    assert len(y_train) == 15
    assert len(y_test) == 5

    assert TARGET not in X_train.columns
    assert TARGET not in X_test.columns


def test_split_data_is_reproducible():

    df = pd.DataFrame({TARGET: np.random.uniform(200, 500, size = 20), "A": np.arange(20, dtype=float)})
    first_split = split_data(df, test_size=0.20, random_state=RANDOM_STATE)
    second_split = split_data(df, test_size=0.20, random_state=RANDOM_STATE)

    pd.testing.assert_frame_equal(first_split[0], second_split[0])  # X_train
    pd.testing.assert_frame_equal(first_split[1], second_split[1])  # X_test
    pd.testing.assert_series_equal(first_split[2], second_split[2]) # y_train
    pd.testing.assert_series_equal(first_split[3], second_split[3]) # y_test

def test_split_data_raises_error_when_target_is_missing():

    df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="Target variable"):
        split_data(df)


def test_prepare_data_for_splitting_caps_mmins():
    
    df = create_minimal_raw_dataframe()
    raw_df, result = prepare_data_for_splitting(df, threshold=100.0)
    assert result["MMINS"].max() == MMINS_CAP


def test_get_feature_groups_separates_variable_types():

    X_train = pd.DataFrame({"MATHEFF": [0.1, 0.2], "MMINS": [200.0, 250.0], "ST004D01T": ["Female", "Male"], "SCHLTYPE": ["Public", "Private"], "ST062Q01TA": [1, 2], "REPEAT": [0, 1], "IMMIG": [1, 2]})

    numeric, categorical, ordinal = get_feature_groups(X_train)

    assert numeric == ["MATHEFF", "MMINS"]
    assert categorical == ["ST004D01T", "SCHLTYPE"]
    assert ordinal == ["ST062Q01TA", "REPEAT", "IMMIG"]

def test_eda_preprocessor_imputes_all_missing_values():

    X_train = pd.DataFrame({"MATHEFF": [0.1, np.nan, 0.3], "ST004D01T": ["Female", np.nan, "Male"], "SCHLTYPE": ["Public", "Public", np.nan], "ST062Q01TA": [1.0, np.nan, 3.0], "REPEAT": [0.0, 1.0, np.nan],"IMMIG": [1.0, 2.0, np.nan]})

    preprocessor = create_eda_preprocessor(X_train)
    transformed = preprocessor.fit_transform(X_train)

    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any()

def test_eda_preprocessor_keeps_all_nominal_categories():

    X_train = pd.DataFrame({
        "MATHEFF": [0.1, 0.2, 0.3, 0.4],
        "ST004D01T": ["Female", "Male", "Female", "Male"],
        "SCHLTYPE": ["Public", "Private", "Public", "Private"],
        "ST062Q01TA": [1, 2, 3, 4],
        "REPEAT": [0, 0, 1, 0],
        "IMMIG": [1, 2, 3, 1]
    })

    preprocessor = create_eda_preprocessor(X_train)
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out().tolist()
    ST004D01T_values = [name for name in feature_names if "ST004D01T" in name]
    SCHLTYPE_values = [name for name in feature_names if "SCHLTYPE" in name]
    assert len(ST004D01T_values) == 2
    assert len(SCHLTYPE_values) == 2

def test_model_imputer_preprocessor_drops_one_reference_category():

    X_train = pd.DataFrame({
        "MATHEFF": [0.1, 0.2, 0.3, 0.4],
        "ST004D01T": ["Female", "Male", "Female", "Male"],
        "SCHLTYPE": ["Public", "Private", "Public", "Private"],
        "ST062Q01TA": [1, 2, 3, 4],
        "REPEAT": [0, 0, 1, 0],
        "IMMIG": [1, 2, 3, 1]
    })

    preprocessor = create_imputer_preprocessor(X_train)
    preprocessor.fit(X_train)
    feature_names = preprocessor.get_feature_names_out().tolist()
    ST004D01T_values = [name for name in feature_names if "ST004D01T" in name]
    SCHLTYPE_values = [name for name in feature_names if "SCHLTYPE" in name]
    assert len(ST004D01T_values) == 1
    assert len(SCHLTYPE_values) == 1

def test_create_imputed_dataframe_appends_target():

    # Use non-default index [10, 11, 12] to verify that the function
    # correctly preserves row alignment after transformation.
    # REFERENCE: https://pandas.pydata.org/docs/user_guide/indexing.html 
    X_train = pd.DataFrame({
        "MATHEFF": [0.1, 0.2, 0.3],
        "ST004D01T": ["Female", "Male", "Female"],
        "SCHLTYPE": ["Public", "Private", "Public"],
        "ST062Q01TA": [1, 2, 3],
        "REPEAT": [0, 1, 0],
        "IMMIG": [1, 2, 3]
    }, index=[10, 11, 12])

    # Same custom index ensures target values stay aligned with their rows
    y = pd.Series([500.0, 550.0, 600.0], index=[10, 11, 12], name=TARGET)

    preprocessor = create_eda_preprocessor(X_train)
    transformed = preprocessor.fit_transform(X_train)

    result = create_imputed_dataframe(transformed, preprocessor, y)

    assert TARGET in result.columns
    assert result[TARGET].tolist() == [500.0, 550.0, 600.0]
    assert len(result) == 3