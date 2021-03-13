import pandas as pd
import numpy as np


class imputation:
    """
    Generates an instance of an imputation class for imputation on missing data

    Attributes
    ----------
    method: str
        method we wish to do the imputing.
    values: numpy array
        an array with values to be imputed. Default None

    Returns
    --------
    An instance of the imputation class

    Examples
    --------
    >>>test_df = pd.DataFrame([[np.nan,2,3],[2,np.nan,4],[5,6,7]])
    >>>imputer = imputation('mean')
    """

    def __init__(self, method):
        """
        Initialize the class

        Parameters
        ----------
        method: str
            method we wish to do the imputing.
        """
        if method not in ["mean", "median", "most_frequent"]:
            raise KeyError("Method must be one of mean, median, most_frequent")
        self.method = method
        self.values = None

    def fit(self, data):
        """
        Calculates the value to be imputated for each column

        Parameters
        ----------
        data: pandas.core.frame.DataFrame
            a pandas dataframe that will be used to compute the values for imputation

        Returns
        -------
        An instance of the imputation class

        Examples
        --------
        >>>test_df = pd.DataFrame([[np.nan,2,3],[2,np.nan,4],[5,6,7]])
        >>>imputer = imputation('mean')
        >>>imputer.fit(test_df)
        """
        if type(data) != pd.DataFrame:
            raise TypeError("Input data must be a Pandas Dataframe")
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
        if self.method == "mean":
            if data.shape[1] != data.select_dtypes(include=np.number).shape[1]:
                raise TypeError("All values in dataframe must be numeric")
            self.values = data.mean().values
        elif self.method == "median":
            if data.shape[1] != data.select_dtypes(include=np.number).shape[1]:
                raise TypeError("All values in dataframe must be numeric")
            self.values = data.median().values
        elif self.method == "most_frequent":
            self.values = data.mode().values[0]

    def fill(self, data_for_fill):
        """
        Fills the missing values in each column

        Parameters
        ----------
        data_for_fill: pandas.core.frame.DataFrame
            a pandas dataframe that we wish to fill the missing values with

        Returns
        -------
        A dataframe with the missing values imputed

        Examples
        --------
        >>>test_df = pd.DataFrame([[np.nan,2,3],[2,np.nan,4],[5,6,7]])
        >>>imputer = imputation('mean')
        >>>imputer.fit(test_df)
        >>>new = imputer.fill(test_df)
        >>>print(new)
               0  1   2
          0  3.5  2.  3.
          1  2.0  4.  4.
          2  5.0  6.  7.
        >>>test_df2 = pd.DataFrame([[1,10,8],[5,2,6],[np.nan,3,np.nan]])
        >>>new2 = imputer.fill(test_df2)
        >>>print(new2)
             0   1         2
        0  1.0  10.  8.000000
        1  5.0   2.  6.000000
        2  3.5   3.  4.666667
        """
        if len(self.values) != data_for_fill.shape[1]:
            raise TypeError(
                "DataFrame to be imputed must have the same number of columns as the fitted data"
            )
        data = data_for_fill.copy()
        for i in range(len(data.columns)):
            data.iloc[:, i].fillna(self.values[i], inplace=True)
        return data
