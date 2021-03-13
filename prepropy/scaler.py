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
        dict:
            dict containing three dataframes with scaled features

        Examples
        ------
        >>>X_train = pd.DataFrame(np.array([['adam', 54, 500],
         ['eve', 45, 6000],['pandaman', 64, 9000]]),
        columns=['name', 'age', 'net_worth'])
        >>>X_Valid = pd.DataFrame(np.array([['nurse', 54, 18000],
         ['ddoorman', 87, 2000],
         ['bruman', 100, 400000]]),
        columns=['name', 'age', 'net_worth'])
        >>>X_test = pd.DataFrame(np.array([['raconman', 45, 70000],
         ['idkman', 23, 56000],
         ['testman', 12, 81000]]),
        columns=['name', 'age', 'net_worth'])
        >>>scaled_data = scaler(X_train,
         X_Valid, X_test,
        ['age', 'net_worth'])
        >>>scaled_data
            {'X_train':        name       age net_worth
             0      adam -0.042954 -1.325838
             1       eve -1.202703  0.236757
             2  pandaman  1.245657  1.089082,
             'X_Valid':        name       age   net_worth
             0     nurse -0.042954    3.646056
             1  ddoorman   4.20946   -0.899676
             2    bruman  5.884654  112.175401,
             'X_test':        name       age  net_worth
             0  raconman -1.202703  18.419684
             1    idkman -4.037646  14.442169
             2   testman -5.455117  21.544874}
    """

    # Error Checking
    if scaler_type not in ["StandardScaler", "MinMaxScaler", "MaxAbsScaler"]:
        raise KeyError(
            'Please select a Valid scaler, pick from "StandardScaler", "MinMaxScaler", "MaxAbsScaler"'
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
