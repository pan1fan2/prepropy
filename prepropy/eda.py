import pandas as pd
import altair as alt
from collections import defaultdict


def eda(df):
    """
    Generates a dictionary to access summary statistics of the given data frame.
    
    Parameters
    ----------
    input_data: pandas.DataFrame
        input dataframe to be analyzed
    Returns
    -------
    dict
        access summary statistics of the given data frame.
    cor
        the correlation map
    Examples
    --------
    >>> from propropy import eda
    >>> df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", ";")
    >>> res = eda(input_data)
    >>> res
    {'nb_missing_values' : 0 ,
     'nb_cat_features' : 0,
     'cat_fetures_name' : []
     'nb_num_features' : 12,
     'num_features_name' : ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
     'class_imbalance' : Yes}

    # the res will also include a correlation map of numerical features
    """