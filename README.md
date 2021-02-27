# prepropy 

![](https://github.com/ja.chang0628/prepropy/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/ja.chang0628/prepropy/branch/main/graph/badge.svg)](https://codecov.io/gh/ja.chang0628/prepropy) ![Release](https://github.com/ja.chang0628/prepropy/workflows/Release/badge.svg) [![Documentation Status](https://readthedocs.org/projects/prepropy/badge/?version=latest)](https://prepropy.readthedocs.io/en/latest/?badge=latest)

A python package for data preprocessing 

## Overview

Data preprocessing and EDA are essential to any data science project. EDA provides insights into a dataset , visualizes and interprets the information that is hidden in the dataset. Data preprocessing is crucial to get rid of any undesired features and handle missing values. In the real world, datasets contain a large number of features and it is unrealistic to expect that raw dataset is perfect and ready for run.The package aims to facilitate users to perform data imputation, feature scaling and data info summaries for machine learning modeling.

## Installation

```bash
$ pip install -i https://test.pypi.org/simple/ prepropy
```

## Features

The package is under development, it will includes following functions:

- Function 1 :  Identify and handle missing values in a dataframe
- Feature Scaler:  Performs Numerical Feature Scaling 
    - Scale Numerical Features to facilitate seamless building of machine learning pipelines
    - Provide functionality to pick from multiple scaling alorithms
- Function 3 :  Extract info and Visualize selected features in a dataframe
    - Separate data into train/test dataset
    - Report number of missing data
    - Report feature types (numerical V.S. categorical)
    - Report class imbalance 
    - Investigate the correlation matrix

EDA is a crucial step to take before diving into any machine learning models. Open-source libraries such as sklearn and pandas provide functions to perform data splitting and data imputation etc. We are not reinventing the functions but we want to integrate function across the packages and provide a quick overview of the data to users.  We hope the package can speed up the data analysis.

## Dependencies

- TODO

## Usage

- TODO

## Documentation

The official documentation is hosted on Read the Docs: https://prepropy.readthedocs.io/en/latest/

## Contributors

|Team Members    | GitHub Username|
|---------------------|-----------|
|Jason Chang | [jachang0628](https://github.com/jachang0628)|
|Bruhat Musunuru | [BruhatM](https://github.com/BruhatM)     |
|Pan Fan       | [pan1fan2](https://github.com/pan1fan2) |

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/ja.chang0628/prepropy/graphs/contributors).

### Credits

This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
