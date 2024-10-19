"""
This is a boilerplate pipeline 'data_mono'
generated using Kedro 0.19.9
"""
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import opendatasets as od
import wquantiles
import os

from pathlib import Path
from scipy.stats import trim_mean
from statsmodels import robust
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.impute import KNNImputer

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.iloc[0]  # Renombrar las columnas con los valores de la fila 0
    df = df.drop(df.index[0])  # Eliminar la primera fila que ahora es innecesaria
    return df

# Función para transformar valores de la columna 'Sexo'
def transform_sexo_column(df: pd.DataFrame) -> pd.DataFrame:
    df['Sexo'] = df['Sexo'].replace({'H': 1, 'M': 2})  # Reemplazar H por 1 y M por 2
    return df

def pre_proceso(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_columns(df)
    df = transform_sexo_column(df)
    return df
