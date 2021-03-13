from prepropy.scaler import scaler
import pytest
import pandas as pd
import numpy as np

# DataFrames used for testing
empty_df = pd.DataFrame()
X_test_list = [["adam", 54, 500], ["eve", 45, 6000], ["pandaman", 64, 9000]]
X_test_non_num = pd.DataFrame(
    np.array(
        [["adam", 54, 500], ["eve", "Iamnotnum", 6000], ["pandaman", 64, 9000]]
    ),
    columns=["name", "age", "net_worth"],
)
X_train = pd.DataFrame(
    np.array([["adam", 54, 500], ["eve", 45, 6000], ["pandaman", 64, 9000]]),
    columns=["name", "age", "net_worth"],
)
X_Valid = pd.DataFrame(
    np.array(
        [["nurse", 54, 18000], ["ddoorman", 87, 2000], ["bruman", 100, 400000]]
    ),
    columns=["name", "age", "net_worth"],
)
X_test = pd.DataFrame(
    np.array(
        [
            ["raconman", 45, 70000],
            ["idkman", 23, 56000],
            ["testman", 12, 81000],
        ]
    ),
    columns=["name", "age", "net_worth"],
)
scaled_data = scaler(X_train, X_Valid, X_test, ["age", "net_worth"])

scaled_max_abs_X_train = np.array(
    [
        [-0.04295367795875608, -1.3258384266690595],
        [-1.202702982845162, 0.2367568619051891],
        [1.2456566608039172, 1.08908156476387],
    ]
)

scaled_max_abs_X_test = np.array(
    [
        [-1.202702982845162, 18.41968385622372],
        [-4.037645728123043, 14.442168576216542],
        [-5.455117100761984, 21.544874433372215],
    ]
)

scaled_max_abs_X_Valid = np.array(
    [
        [-0.04295367795875608, 3.646055673339913],
        [4.2094604399580655, -0.899676075239719],
        [5.884653880349541, 112.17540117067863],
    ]
)


# testing if output is correct
def test_scaled_values():
    """Tests if the scaled values are correct"""
    temp = scaler(X_train, X_Valid, X_test, ["age", "net_worth"])
    assert np.array_equal(
        temp["X_test"].drop(columns=["name"]).to_numpy(), scaled_max_abs_X_test
    ), "Incorrect scaled test values"
    assert np.array_equal(
        temp["X_train"].drop(columns=["name"]).to_numpy(),
        scaled_max_abs_X_train,
    ), "Incorrect scaled train values"
    assert np.array_equal(
        temp["X_Valid"].drop(columns=["name"]).to_numpy(),
        scaled_max_abs_X_Valid,
    ), "Incorrect scaled Valid values"


# testing for type and value errors
def test_empty():
    """Tests whether function catches empty dataframe"""
    with pytest.raises(ValueError):
        scaler(empty_df, empty_df, empty_df, ["col_1", "col_2"])


def test_scaling_method():
    """Tests whether scaling Method catches invalid methods"""
    with pytest.raises(KeyError):
        scaler(
            X_train,
            X_Valid,
            X_test,
            ["age", "net_worth"],
            scaler_type="spacescaler",
        )


def test_input_type():
    """Tests if function catches wrong data type"""
    with pytest.raises(TypeError):
        scaler(X_train, X_Valid, X_test_list, ["age", "net_worth"])


def test_input_datatypes():
    """Tests if data in the given dataframe is of correct datatype"""
    with pytest.raises(ValueError):
        scaler(X_train, X_Valid, X_test_non_num, ["age", "net_worth"])
