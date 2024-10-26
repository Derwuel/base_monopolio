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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger(__name__)


def split_data2(df: pd.DataFrame, parameters: t.Dict) -> t.Tuple:
    X = df[parameters["featrures01"]]
    y = df["Renta"]  # Asegúrate de que 'Renta' sea una variable categórica
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_nume"])
    return X_train, X_test, y_train, y_test

def mod_SGDClassifier(X_train: pd.DataFrame, y_train: pd.Series) -> SGDClassifier:
    model_04 = SGDClassifier()
    model_04.fit(X_train, y_train)
    return model_04

def evaluacion_SGDClassifier(model_04: SGDClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred_04 = model_04.predict(X_test)
    accuracy_04 = accuracy_score(y_test, y_pred_04)
    logger.info("Accuracy: %s", accuracy_04)
    return accuracy_04

def mod_tree_model_clasicicacion(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    tree_model_clasicicacion = DecisionTreeClassifier(max_depth=10, random_state=42)
    tree_model_clasicicacion.fit(X_train, y_train)
    return tree_model_clasicicacion

def evaluacion_tree_model_clasicicacion(tree_model_clasicicacion: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred_05 = tree_model_clasicicacion.predict(X_test)
    accuracy_05 = accuracy_score(y_test, y_pred_05)
    report_05 = classification_report(y_test, y_pred_05)
    logger.info("Accuracy: %s", accuracy_05)
    logger.info("report: %s", report_05)
    return accuracy_05, report_05