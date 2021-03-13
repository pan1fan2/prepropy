import pandas as pd
import altair as alt


def eda(df, target):
    """
    Generates a dictionary to access summary statistics of the given data frame
    
        Parameters
        --------
        df : pandas.DataFrame
            input dataframe to be analyzed
        target : string
            target column name
        
        Returns
        --------
        dict
            access summary statistics of the given data frame.
        cor
            the correlation map
        
        Examples
        --------
        >>> from propropy import eda
        >>> url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        >>> url2 = "wine-quality/winequality-red.csv"
        >>> url = url1+url2
        >>> df = pd.read_csv(url, ";")
        >>> target = "quality"
        >>> res = eda(df,quality)
        >>> res
        {'nb_missing_values': [('fixed acidity', 0),
        ('volatile acidity', 0),
        ('citric acid', 0),
        ('residual sugar', 0),
        ('chlorides', 0),
        ('free sulfur dioxide', 0),
        ('total sulfur dioxide', 0),
        ('density', 0),
        ('pH', 0),
        ('sulphates', 0),
        ('alcohol', 0)],
        'nb_cat_features': 0,
        'cat_features_name': [],
        'nb_num_features': 11,
        'num_features_name': ['fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'],
        'nb_class': 6,
        'class_ratio': [0.4259, 0.399, 0.1245, 0.0331, 0.0113, 0.0063],
        'pairplot': alt.RepeatChart(...)}
    """
    # Check the dataframe input
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be an instance of DataFrame")
    # Create an empty dictionary
    res = {}
    # obtain statistical information
    df_fea = df.drop(target, 1)
    num_fea = df_fea.select_dtypes('number').columns.to_list()
    cat_fea = list(set(list(df_fea.columns)) - set(num_fea))
    key_null = list(df_fea.isnull().sum().index)
    val_null = list(df_fea.isnull().sum().values)
    res["nb_missing_values"] = list(zip(key_null, val_null))
    res["nb_cat_features"] = len(cat_fea)
    res['cat_features_name'] = cat_fea
    res["nb_num_features"] = len(num_fea)
    res['num_features_name'] = num_fea
    res["nb_class"] = len(list(set(df[target])))
    class_count = df[target].value_counts(normalize=True).values
    res["class_ratio"] = list(class_count.round(4))
    # Create a pair plot with Altair
    color_lab = target + ":N"
    chart = alt.Chart(df).mark_circle().encode(
        alt.X(alt.repeat("column"), type='quantitative'),
        alt.Y(alt.repeat("row"), type='quantitative'),
        color=color_lab
    ).properties(
        width=100,
        height=100
    ).repeat(
        row=num_fea,
        column=num_fea
    )
    res["pairplot"] = chart
    return res
