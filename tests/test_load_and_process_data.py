import numpy as np
import pandas as pd
import pytest
import pytest_mock

from src.load_and_process_data import connect_redshift, load_data, reduce_memory_usage, remove_outliers, \
    extract_most_common_values, \
    bucket_col_based_on_most_common_values, add_bucked_col_to_df, pre_training_data_processing, make_dummies, \
    data_processing


@pytest.mark.skip
def test_connect_redshift():
    pass


@pytest.mark.skip
def test_load_data():
    pass


def test_reduce_memory_usage(input_df):
    initial_df = input_df.copy()
    initial_df_memory_usage = initial_df.memory_usage(deep=True).sum()
    initial_df_cols_sum = sum(initial_df.sum(numeric_only=True))

    expected_df = reduce_memory_usage(df=input_df.copy(), int_cast=True, float_cast=False, obj_to_category=False)
    expected_df_memory_usage = expected_df.memory_usage(deep=True).sum()
    expected_df_cols_sum = sum(expected_df.sum(numeric_only=True))

    assert initial_df_memory_usage > expected_df_memory_usage
    assert initial_df_cols_sum == expected_df_cols_sum


@pytest.mark.parametrize("feature, threshold, is_above, expected_output",
                         [("feature_3", 1000, True, True),
                          ("feature_3", np.inf, True, False)])
def test_remove_outliers(input_df, feature, threshold, is_above, expected_output):
    initial_df_len = len(input_df.index)

    expected_df = remove_outliers(df=input_df.copy(), feature=feature, threshold=threshold, is_above=is_above)
    expected_df_len = len(expected_df.index)

    assert (initial_df_len > expected_df_len) == expected_output


@pytest.mark.parametrize("feature, top_x, expected_output",
                         [("feature_4", 1, True),
                          ("feature_4", 5, False)])
def test_extract_most_common_values(input_df, feature, top_x, expected_output):
    initial_most_common_values = list(input_df[feature].mode())

    expected_most_common_values = extract_most_common_values(df=input_df.copy(), feature=feature, top_x=top_x)

    assert (initial_most_common_values == expected_most_common_values) == expected_output


@pytest.mark.parametrize("feature, most_common_values",
                         [("feature_4", ["a", "b"])])
def test_bucket_col_based_on_most_common_values(input_df, feature, most_common_values):
    initial_df = input_df.copy()
    initial_feature_bucket = list(np.where(initial_df[feature].isin(most_common_values), initial_df[feature], "Other"))

    expected_df = input_df.copy()
    expected_feature_bucket = list(expected_df.apply(
        lambda row: bucket_col_based_on_most_common_values(row=row,
                                                           feature=feature,
                                                           most_common_values=most_common_values), axis=1))

    assert initial_feature_bucket == expected_feature_bucket


def test_add_bucked_col_to_df(input_df, mocker):
    mocker.patch("src.load_and_process_data.extract_most_common_values", return_value=None)
    mocker.patch("src.load_and_process_data.bucket_col_based_on_most_common_values", return_value="Other")

    initial_df = input_df.copy()
    initial_df["feature_4_bucket"] = "Other"

    expected_df = add_bucked_col_to_df(df=input_df.copy(), feature="feature_4", top_x=None)

    assert initial_df.equals(expected_df)


@pytest.mark.skip
def test_pre_training_data_processing():
    pass


@pytest.mark.skip
def test_make_dummies():
    pass


@pytest.mark.skip
def test_data_processing():
    pass
