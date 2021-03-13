from prepropy.imputation import imputation
import pytest
import pandas as pd
import numpy as np


def test_imputation():
    """Group of tests that test for the output generated from impute"""
    df1 = pd.DataFrame([[np.nan, 2, 3], [4, np.nan, 6], [10, 5, 9]])
    df2 = pd.DataFrame(
        [[np.nan, 2, 3], [4, np.nan, 6], [10, 5, 9], [3, 15, 17]]
    )
    df3 = pd.DataFrame(
        [[np.nan, "b", "c"], ["d", np.nan, "f"], ["d", "x", np.nan]]
    )
    df_m = pd.DataFrame(
        [[np.nan, 1, "c"], ["d", np.nan, "f"], ["d", 3, np.nan]]
    )
    df_s = pd.DataFrame([1])
    df1_t = pd.DataFrame([[7.0, 2, 3], [4, 3.5, 6], [10, 5, 9]])
    df2_t = pd.DataFrame(
        [[4.0, 2.0, 3.0], [4.0, 5.0, 6.0], [10.0, 5.0, 9.0], [3.0, 15.0, 17.0]]
    )
    df3_t = pd.DataFrame([["d", "b", "c"], ["d", "b", "f"], ["d", "x", "c"]])
    df_m_t = pd.DataFrame([["d", 1.0, "c"], ["d", 1.0, "f"], ["d", 3.0, "c"]])
    # test for mean imputation
    imputer = imputation("mean")
    imputer.fit(df1)
    df1_filled = imputer.fill(df1)
    assert np.array_equal(
        df1_filled.to_numpy(), df1_t.to_numpy()
    ), "Generated Output are incorrect"

    # test for median imputation
    imputer = imputation("median")
    imputer.fit(df2)
    df2_filled = imputer.fill(df2)
    assert np.array_equal(
        df2_filled.to_numpy(), df2_t.to_numpy()
    ), "Generated Output are incorrect"

    # test for most frequent imputation
    imputer = imputation("most_frequent")
    imputer.fit(df3)
    df3_filled = imputer.fill(df3)
    assert np.array_equal(
        df3_filled.to_numpy(), df3_t.to_numpy()
    ), "Generated Output are incorrect"

    # test for dataframe with non nan values
    imputer = imputation("mean")
    imputer.fit(df1_t)
    df4_filled = imputer.fill(df1_t)
    assert np.array_equal(
        df4_filled.to_numpy(), df1_t.to_numpy()
    ), "Generated Output are incorrect"

    # test for mix type dataframe for most_frequent
    imputer = imputation("most_frequent")
    imputer.fit(df_m)
    df5_filled = imputer.fill(df_m)
    assert np.array_equal(
        df5_filled.to_numpy(), df_m_t.to_numpy()
    ), "Generated Output are incorrect"

    # test for dataframe where there is only a single value
    imputer = imputation("mean")
    imputer.fit(df_s)
    df6_filled = imputer.fill(df_s)
    assert np.array_equal(
        df6_filled.to_numpy(), df_s.to_numpy()
    ), "Generated Output are incorrect"


# Exception Handling
def test_empty():
    """Tests whether Class Method catches empty dataframe"""
    emp = pd.DataFrame()
    imputer = imputation("mean")
    with pytest.raises(ValueError):
        imputer.fit(emp)


def test_method():
    """Tests catches wrong method for imputation"""
    with pytest.raises(KeyError):
        imputer = imputation("hello")  # noqa: F841


def test_pdtype():
    """Tests whether Class Method catches wrong datatype"""
    with pytest.raises(TypeError):
        imputer = imputation("mean")
        imputer.fit(1)


def test_pdtype2():
    """Tests given mean/median, df is all numeric"""
    test_df = pd.DataFrame(
        [["a", "b", "c"], ["d", np.nan, "f"], ["z", "x", "y"]]
    )
    with pytest.raises(TypeError):
        imputer = imputation("mean")
        imputer.fit(test_df)
    with pytest.raises(TypeError):
        imputer = imputation("median")
        imputer.fit(test_df)


def test_length():
    """Tests whether columns of fit = fill"""
    test_df1 = pd.DataFrame(
        [["a", "b", "c"], ["d", np.nan, "f"], ["z", "x", "y"]]
    )
    test_df2 = pd.DataFrame([["a", "b"], ["d", np.nan], ["z", "x"]])
    with pytest.raises(TypeError):
        imputer = imputation("most_frequent")
        imputer.fit(test_df1)
        imputer.fill(test_df2)
