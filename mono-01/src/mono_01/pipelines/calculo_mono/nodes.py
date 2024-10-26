"""
This is a boilerplate pipeline 'calculo_mono'
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)

def split_data2(df: pd.DataFrame, parameters: t.Dict) -> t.Tuple:
    X = df[parameters["featrures01"]]
    y = df["Renta"]  # Asegúrate de que 'Renta' sea una variable categórica
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_nume"])
    return X_train, X_test, y_train, y_test

def mod_regresion_logistica(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model_4 = LogisticRegression(max_iter=1000)  # Aumentar iteraciones si es necesario
    model_4.fit(X_train, y_train)
    return model_4

def evaluacion_regresion_logistica(model_4: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model_4.predict(X_test)
    accuracy_4 = accuracy_score(y_test, y_pred)
    f1_4 = f1_score(y_test, y_pred, average='weighted')
    logger.info("Accuracy: %s", accuracy_4)
    logger.info("F1 Score: %s", f1_4)
    return accuracy_4, f1_4

def mod_arbol_decision(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    model_5 = DecisionTreeClassifier(random_state=30, max_depth=5)
    model_5.fit(X_train, y_train)
    return model_5

def evaluacion_arbol_decision(model_5: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model_5.predict(X_test)
    accuracy_5 = accuracy_score(y_test, y_pred)
    f1_5 = f1_score(y_test, y_pred, average='weighted')
    logger.info("Accuracy: %s", accuracy_5)
    logger.info("F1 Score: %s", f1_5)
    return accuracy_5, f1_5

def mod_svc(X_train: pd.DataFrame, y_train: pd.Series) -> SVC:
    model_6 = SVC(kernel='rbf')
    model_6.fit(X_train, y_train)
    return model_6

def evaluacion_svc(model_6: SVC, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model_6.predict(X_test)
    accuracy_6 = accuracy_score(y_test, y_pred)
    f1_6 = f1_score(y_test, y_pred, average='weighted')
    logger.info("Accuracy: %s", accuracy_6)
    logger.info("F1 Score: %s", f1_6)
    return accuracy_6, f1_6