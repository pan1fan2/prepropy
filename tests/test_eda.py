import numpy as np
import pandas as pd
import altair as alt
import pytest
from prepropy import eda

def test_eda():
    
    def gen_test_data():
        
        test_data = pd.DataFrame()
        test_data["num1"] = [8.5, 8, 9.2, 9.1, 9.4]
        test_data["num2"] = [0.88, 0.93, 0.95 , 0.92 , 0.91]
        test_data["num3"] = [0.46, 0.78, 0.66, 0.69, 0.52]
        test_data["num4"] = [0.082, 0.078, 0.082, 0.085, 0.066]
        test_data["cat1"] = ["Good","Okay","Excellent","Terrible","Good"]
        test_data["target"] = [2,2,3,1,3]
        
        return test_data
    
    # test for attribute names
    def test_attributes():
        
        test_data = gen_test_data()
        eda_res = eda.eda(test_data)

        assert 'cat_features_name' in eda_res
        assert 'num_features_name' in eda_res
        assert 'pairplot' in eda_res
        
    # test for numerical features   
    def test_num_feature():
        
        test_data = gen_test_data()
        eda_res = eda.eda(test_data)
        
        assert "num2" in eda_res["num_features_name"] , "Wrong output for numerical feature names"
    
    # test for length of numerical features
    def test_len_num_feature():
        
        test_data = gen_test_data()
        eda_res = eda.eda(test_data)
        
        assert eda_res["nb_num_features"] == 4 , "Wrong output for the length of numerical features"
    
    # test for length of numerical features
    def test_cat_feature():
        
        test_data = gen_test_data()
        eda_res = eda.eda(test_data)
        
        assert "cat1" in eda_res["cat_features_name"] , "Wrong output for categorical feature names"
    
    # test for length of categorical features
    def test_len_cat_features():
        
        test_data = gen_test_data()
        eda_res = eda.eda(test_data)
        
        assert eda_res["nb_cat_features"] == 1, "Wrong output for the length of categorical features"
        
    
    # test for the number of class labels
    
    def test_nb_classes():
        
        test_data = gen_test_data()
        eda_res = eda.eda(test_data)
        
        assert eda_res["nb_class"] == 3 , "Wrong output for the number of class lables"

    # test for altair output
    def test_pairplot():
    
        test_data = gen_test_data()
        eda_res = eda.eda(test_data)
        p = eda_res["pairplot"]
        assert isinstance(p, altair.vegalite.v4.api.RepeatChart), 'Wrong output for altair plot'
        
    def test_eda_raise_error_not_dataframe():
        test_input = [1, 2, 3, 4, 5]

        with pytest.raises(ValueError, match="input_data must be instance of pandas.core.frame.DataFrame."):
            eda.eda(test_input) 
        