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

def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lista de columnas que deseas conservar
    columns_to_keep = ['Id', 'Subsegmento', 'Sexo', 'Region', 'Edad', 'Renta', 'Antiguedad', 
                       'Internauta', 'Adicional', 'Dualidad', 'Monoproducto', 'Ctacte',
                       'Consumo', 'Hipotecario', 'Debito', 'CambioPin', 'Cuentas', 'TC']
    # Filtrar el DataFrame manteniendo solo las columnas que están en la lista
    df_filtered = df[columns_to_keep]
    return df_filtered

def imputar_datos(df_filtered: pd.DataFrame) -> pd.DataFrame:
    # Definir las columnas para la imputación
    columnas_para_imputar = ['Id', 'Subsegmento', 'Sexo', 'Region', 'Edad', 'Renta', 'Antiguedad',
                             'Internauta', 'Adicional', 'Dualidad', 'Monoproducto', 'Ctacte',
                             'Consumo', 'Hipotecario', 'Debito', 'CambioPin', 'Cuentas', 'TC']
    # Convertir las columnas seleccionadas a float32 para la imputación
    df_filtered[columnas_para_imputar] = df_filtered[columnas_para_imputar].astype('float32')
    # Configurar el imputador KNN con 5 vecinos
    imputer = KNNImputer(n_neighbors=5)
    # Realizar la imputación en las columnas seleccionadas
    df_filtered_imputed = pd.DataFrame(imputer.fit_transform(df_filtered[columnas_para_imputar]), 
                                       columns=columnas_para_imputar)
    # Reemplazar las columnas imputadas en el DataFrame original filtrado
    df_filtered[columnas_para_imputar] = df_filtered_imputed
    # Convertir todas las columnas imputadas a int64
    df_filtered[columnas_para_imputar] = df_filtered[columnas_para_imputar].fillna(0).astype('int32')
    return df_filtered

def pre_proceso(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_columns(df)
    df = transform_sexo_column(df)
    df_filtered = filter_columns(df)
    df_filtered = imputar_datos(df_filtered)
    return df_filtered
