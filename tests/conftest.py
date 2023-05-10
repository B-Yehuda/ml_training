import pandas as pd
import pytest


@pytest.fixture
def input_df():
    df = pd.DataFrame({'feature_1': [10, 20, 30, 40],
                       'feature_2': [100, 200, 300, 400],
                       'feature_3': [1000, 2000, 3000, 4000],
                       'feature_4': ["a", "b", "c", "a"]
                       })

    df[['feature_1', 'feature_2', 'feature_3']].astype('int64')

    return df
