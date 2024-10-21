"""
This is a boilerplate pipeline 'modelo_mono'
generated using Kedro 0.19.9
"""
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import opendatasets as od
import sklearn
import wquantiles
import os
import typing as t
import logging

from pathlib import Path
from scipy.stats import trim_mean
from statsmodels import robust
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)

def split_data(df: pd.DataFrame, parameters: t.Dict) -> t.Tuple:
    X = df[parameters["featrures01"]]
    y = df["Renta"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_nume"])
    return X_train, X_test, y_train, y_test

def mod_regrecion_lineal(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluacion_regrecion_lineal(regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    logger.info("este es el resultado de la evaluacion: ", r2)