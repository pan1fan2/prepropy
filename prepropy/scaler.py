import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler


def scaler(
    X_train, X_Valid, X_test, scale_features, scaler_type="StandardScaler"
):
    """
    This function scales numerical features based on scaling requirement

    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame, numpy array or list
        The DataFrame, numpy array or list
    X_Valid : pandas.core.frame.DataFrame, numpy array or list
        The DataFrame, numpy array or list
    X_test : pandas.core.frame.DataFrame, numpy array or list
        The DataFrame, numpy array or list
    scale_features: list of strings
        The list of numerical features to be scaled
    scaler_type: string
        The type of scaling to perform on the numerical columns.

    Returns
    -------
    dict
        dict containing three dataframes with scaled features

    Examples
    --------
    >>>scaler(X_train, X_Valid, X_test,
     scale_features, scaler_type="MaxAbsScaler")
    """

    # Error Checking
    if scaler_type not in ["StandardScaler", "MinMaxScaler", "MaxAbsScaler"]:
        raise KeyError(
            'Please use scaler "StandardScaler", "MinMaxScaler", "MaxAbsScaler"'  # noqa: E501
        )
    if (
        not isinstance(X_train, pd.DataFrame)
        or not isinstance(X_Valid, pd.DataFrame)
        or not isinstance(X_test, pd.DataFrame)
    ):
        raise TypeError("Input data must be a Pandas Dataframe")
    if (
        X_train.empty
        or X_Valid.empty
        or X_test.empty
        or (len(scale_features) == 0)
    ):
        raise ValueError("Inputs cannot be empty")
    if not isinstance(scale_features, list):
        raise TypeError("Feature names must be in list format")
    for feature in scale_features:
        if (X_train[feature].str.isnumeric().sum()) != len(X_train[feature]):
            raise ValueError("Features should have only numeric values")
        if (X_Valid[feature].str.isnumeric().sum()) != len(X_Valid[feature]):
            raise ValueError("Features should have only numeric values")
        if (X_test[feature].str.isnumeric().sum()) != len(X_test[feature]):
            raise ValueError("Features should have only numeric values")

    # Scaling Instance
    scaled_data = {}
    if scaler_type == "StandardScaler":
        scaler_instance = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler_instance = MinMaxScaler()
    elif scaler_type == "MaxAbsScaler":
        scaler_instance = MaxAbsScaler()

    # Fitting the data for Scaling
    scaler_instance.fit(X_train[scale_features])

    # Scaling train data
    X_train_scaled = X_train.copy()
    X_train_scaled[scale_features] = scaler_instance.transform(
        X_train[scale_features]
    )
    scaled_data["X_train"] = X_train_scaled

    # Scaling the validation data
    X_Valid_scaled = X_Valid.copy()
    X_Valid_scaled[scale_features] = scaler_instance.transform(
        X_Valid[scale_features]
    )
    scaled_data["X_Valid"] = X_Valid_scaled

    # Scaling the test data
    X_test_scaled = X_test.copy()
    X_test_scaled[scale_features] = scaler_instance.transform(
        X_test[scale_features]
    )
    scaled_data["X_test"] = X_test_scaled

    return scaled_data
