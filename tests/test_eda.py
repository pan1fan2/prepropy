import pandas as pd
import altair as alt
import pytest
from prepropy import eda


def gen_test_data():
    test_data = pd.DataFrame()
    test_data["num1"] = [8.5, 8, 9.2, 9.1, 9.4]
    test_data["num2"] = [0.88, 0.93, 0.95, 0.92, 0.91]
    test_data["num3"] = [0.46, 0.78, 0.66, 0.69, 0.52]
    test_data["num4"] = [0.082, 0.078, 0.082, 0.085, 0.066]
    test_data["cat1"] = ["Good", "Okay", "Excellent", "Terrible", "Good"]
    test_data["target"] = [2, 2, 3, 1, 3]
    return test_data


# test for attribute names
def test_attributes():
    test_data = gen_test_data()
    eda_res = eda.eda(test_data, "target")
    assert 'cat_features_name' in eda_res
    assert 'num_features_name' in eda_res
    assert 'pairplot' in eda_res


# test for numerical features
def test_num_feature():
    test_data = gen_test_data()
    eda_res = eda.eda(test_data, "target")
    message = "Wrong output for numerical feature names"
    assert "num2" in eda_res["num_features_name"], message


# test for length of numerical features
def test_len_num_feature():
    test_data = gen_test_data()
    eda_res = eda.eda(test_data, "target")
    message = "Wrong output for the length of numerical features"
    assert eda_res["nb_num_features"] == 4, message


# test for length of numerical features
def test_cat_feature():
    test_data = gen_test_data()
    eda_res = eda.eda(test_data, "target")
    message = "Wrong output for categorical feature names"
    assert "cat1" in eda_res["cat_features_name"], message


# test for length of categorical features
def test_len_cat_features():
    test_data = gen_test_data()
    eda_res = eda.eda(test_data, "target")
    message = "Wrong output for the length of categorical features"
    assert eda_res["nb_cat_features"] == 1, message


# test for the number of class labels
def test_nb_classes():
    test_data = gen_test_data()
    eda_res = eda.eda(test_data, "target")
    message = "Wrong output for the number of class lables"
    assert eda_res["nb_class"] == 3, message


# test for altair output
def test_pairplot():
    test_data = gen_test_data()
    eda_res = eda.eda(test_data, "target")
    p = eda_res["pairplot"]
    message = 'Wrong output for altair plot'
    assert isinstance(p, alt.vegalite.v4.api.RepeatChart), message


# Handle exception
def test_eda_raise_error_not_dataframe():
    test_input = [1, 2, 3, 4, 5]
    message = "Input data must be an instance of DataFrame"
    with pytest.raises(TypeError, match=message):
        eda.eda(test_input, "target")
