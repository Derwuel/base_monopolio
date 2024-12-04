# Importar bibliotecas necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

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
    models_data = []
    for path in pickle_paths:
        with open(path, "rb") as f:
            model_data = pickle.load(f)
            models_data.append(model_data)
    return models_data

# Función para preparar métricas de los modelos
def prepare_metrics(models_data):
    """Crear un DataFrame con las métricas de los modelos."""
    metrics_list = []
    for model in models_data:
        try:
            metrics_list.append({
                "Model": model.get("name", "Unknown"),
                "Accuracy": model.get("accuracy", 0),
                "Precision": model.get("precision", 0),
                "Recall": model.get("recall", 0),
                "F1-Score": model.get("f1_score", 0),
            })
        except Exception as e:
            st.warning(f"No se pudieron extraer métricas: {e}")
    return pd.DataFrame(metrics_list)

# Función para preparar el perfil del cliente objetivo
def prepare_client_profile(df):
    """Generar un perfil promedio del cliente objetivo."""
    return df.mean().reset_index(name="Value").rename(columns={"index": "Feature"})

# Ruta del archivo de datos
data_file = "data/02_intermediate/pos_proceso.pq"
csv_output_file = "data/02_intermediate/pos_proceso.csv"

# Leer datos y convertir a CSV si es necesario
if data_file.endswith(".pq"):
    try:
        df = pd.read_parquet(data_file)
        # Guardar como CSV
        os.makedirs(os.path.dirname(csv_output_file), exist_ok=True)
        df.to_csv(csv_output_file, index=False)
        st.info(f"Archivo Parquet convertido a CSV y guardado en: {csv_output_file}")
    except Exception as e:
        st.error(f"Error al leer o convertir el archivo Parquet: {e}")
        df = pd.DataFrame()  # Crear un DataFrame vacío
else:
    try:
        df = pd.read_csv(data_file)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        df = pd.DataFrame()

# Filtrar columnas requeridas
columns_required = ["Sexo", "Región", "Edad", "Antigüedad", "Monoproducto", "Consumo"]
missing_columns = [col for col in columns_required if col not in df.columns]
if missing_columns:
    st.warning(f"Las siguientes columnas faltan en los datos: {missing_columns}")
else:
    df = df[columns_required]

# Preparar métricas y cliente objetivo si el DataFrame no está vacío
if not df.empty:
    pickle_files = [
        "data/06_models/tree_model.pickle",
        "data/06_models/resultado01.pickle",
        "data/06_models/svr_model.pickle",
    ]
    models_data = load_models(pickle_files)
    metrics = prepare_metrics(models_data)
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
