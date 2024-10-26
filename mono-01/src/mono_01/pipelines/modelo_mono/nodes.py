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
    logger.info("este es el resultado de la evaluacion: %s", r2)

def mod_arbol_decision(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeRegressor:
    # Crear y ajustar el modelo de Árbol de Decisiones
    model2 = DecisionTreeRegressor(random_state=42)
    model2.fit(X_train, y_train)
    return model2

def evaluacion_arbol_decision(model: DecisionTreeRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    # Predecir con los datos de prueba
    y_pred = model.predict(X_test)
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Registrar resultados
    logger.info("Mean Squared Error (MSE): %s", mse)
    logger.info("R2 Score: %s", r2)
    return mse, r2

def mod_svr(X_train: pd.DataFrame, y_train: pd.Series) -> SVR:
    # Crear y ajustar el modelo de SVR
    model3 = SVR(kernel='rbf')  # Puedes probar otros kernels como 'linear', 'poly', etc.
    model3.fit(X_train, y_train)
    return model3

def evaluacion_svr(model3: SVR, X_test: pd.DataFrame, y_test: pd.Series):
    # Predecir con los datos de prueba
    y_pred = model3.predict(X_test)
    # Calcular métricas
    mse3 = mean_squared_error(y_test, y_pred)
    r2_3 = r2_score(y_test, y_pred)
    # Registrar resultados
    logger.info("Mean Squared Error (MSE): %s", mse3)
    logger.info("R2 Score: %s", r2_3)
    return mse3, r2_3