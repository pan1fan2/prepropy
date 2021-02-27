import numpy as np
import pandas as pd

def scaler():
    """
    This function scales numerical features based on scaling requirement

        Parameters
        ----------
        X_train : pandas.core.frame.DataFrame, numpy array or list
            The DataFrame, numpy array or list containing training examples that need scaling

        X_Valid : pandas.core.frame.DataFrame, numpy array or list
            The DataFrame, numpy array or list containing Validation examples that need scaling

        X_test : pandas.core.frame.DataFrame, numpy array or list
            The DataFrame, numpy array or list containing testing examples that need scaling

        scale_features: list of strings
            The list of numerical features to be scaled
        
        scaler_type: string
            The type of scaling to perform on the numerical columns. Pick from ["StandardScaler", "MinMaxScaler", "MaxAbsScaler"]
        

        Returns
        -------
        dict: 
            dict containing three dataframes with scaled features
        

        Examples
        ------
            Placeholder
    """