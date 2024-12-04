# Importar bibliotecas necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Configuración de la página
st.set_page_config(
    page_title="Análisis del Cliente Objetivo",
    page_icon="📊",
    layout="wide",
)

# Función para cargar modelos desde archivos pickle
@st.cache_data
def load_models(pickle_paths):
    """Cargar modelos desde archivos pickle."""
    models = []
    feature_names = None
    for path in pickle_paths:
        with open(path, "rb") as f:
            model = pickle.load(f)
            models.append(model)
            # Intentar cargar las características si están presentes
            if hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
    return models, feature_names

# Función para calcular métricas de los modelos
def calculate_metrics(models, X_test, y_test):
    """Calcular métricas para cada modelo."""
    metrics_list = []
    for model in models:
        try:
            # Realizar predicción
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)  # Si es necesario binarizar las predicciones
            # Calcular métricas
            metrics_list.append({
                "Model": type(model).__name__,
                "Accuracy": accuracy_score(y_test, y_pred_binary),
                "Precision": precision_score(y_test, y_pred_binary, zero_division=0),
                "Recall": recall_score(y_test, y_pred_binary, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred_binary, zero_division=0),
            })
        except Exception as e:
            st.warning(f"No se pudieron calcular métricas para el modelo {type(model).__name__}: {e}")
    return pd.DataFrame(metrics_list)

# Función para preparar el perfil del cliente objetivo
def prepare_client_profile(df):
    """Generar un perfil promedio del cliente objetivo."""
    return df.mean().reset_index(name="Value").rename(columns={"index": "Feature"})

# Ruta del archivo de datos
data_file = "data/02_intermediate/pos_proceso.pq"

# Leer datos y ajustar columnas requeridas
try:
    df = pd.read_parquet(data_file)
    st.info("Archivo Parquet cargado correctamente.")
    st.write("Columnas disponibles en el archivo:", df.columns.tolist())
except Exception as e:
    st.error(f"Error al cargar el archivo Parquet: {e}")
    df = pd.DataFrame()

# Preparar métricas y cliente objetivo si el DataFrame no está vacío
if not df.empty:
    # Cargar modelos desde archivos pickle
    pickle_files = [
        "data/06_models/tree_model.pickle",
        "data/06_models/resultado01.pickle",
        "data/06_models/svr_model.pickle",
    ]
    models, feature_names = load_models(pickle_files)

    # Alinear características con las usadas durante el entrenamiento
    if feature_names is not None:
        # Asegurar que solo las columnas necesarias estén presentes
        missing_features = [col for col in feature_names if col not in df.columns]
        extra_features = [col for col in df.columns if col not in feature_names]

        # Agregar columnas faltantes con valores predeterminados
        for col in missing_features:
            df[col] = 0

        # Eliminar columnas adicionales
        df = df[feature_names]

        st.write(f"Características alineadas con los modelos: {df.columns.tolist()}")

    # Dividir datos para evaluar los modelos
    # En este caso, `Sexo` es una característica, no la columna objetivo
    X = df  # Usamos todo el DataFrame como `X`
    y = df["Sexo"]  # Cambiar esto si otra columna es la variable objetivo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calcular métricas
    metrics = calculate_metrics(models, X_test, y_test)

    # Preparar perfil del cliente objetivo
    client_profile = prepare_client_profile(df)

    # Título de la aplicación
    st.title("📊 Resultados del Entrenamiento y Análisis del Cliente Objetivo")

    # Sección 1: Métricas de los modelos
    st.header("📈 Métricas de los Modelos Entrenados")
    st.dataframe(metrics)

    # Gráfico de comparación de precisión
    if st.checkbox("Mostrar gráfico de rendimiento de modelos"):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=metrics, x="Model", y="Accuracy", ax=ax, palette="viridis")
        ax.set_title("Comparación de Precisión entre Modelos")
        ax.set_ylabel("Precisión")
        ax.set_xlabel("Modelo")
        st.pyplot(fig)

    # Sección 2: Cliente objetivo
    st.header("👤 Cliente Objetivo")
    st.write("Características promedio del cliente objetivo:")
    st.dataframe(client_profile)

    # Gráfico de características del cliente objetivo
    if st.checkbox("Mostrar gráfico de distribución de características"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=client_profile, x="Value", y="Feature", ax=ax, palette="coolwarm")
        ax.set_title("Distribución de Características del Cliente Objetivo")
        ax.set_xlabel("Valor Promedio")
        st.pyplot(fig)

    # Gráfico de histograma para una característica clave
    if st.checkbox("Mostrar histograma de una característica clave"):
        feature = st.selectbox("Selecciona una característica para el histograma", df.columns)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[feature], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Histograma de {feature}")
        ax.set_xlabel(feature)
        st.pyplot(fig)

    # Gráfico de dispersión para dos características
    if st.checkbox("Mostrar gráfico de dispersión entre dos características"):
        x_feature = st.selectbox("Selecciona la característica del eje X", df.columns)
        y_feature = st.selectbox("Selecciona la característica del eje Y", df.columns)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_feature, y=y_feature, ax=ax, color="green", alpha=0.7)
        ax.set_title(f"Dispersión entre {x_feature} y {y_feature}")
        st.pyplot(fig)
else:
    st.error("No se pudieron cargar los datos. Revisa los archivos de entrada.")