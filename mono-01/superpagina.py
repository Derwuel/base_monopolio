# Importar bibliotecas necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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
    try:
        profile = pd.DataFrame({
            "Feature": df.columns,
            "Value": df.mean()
        }).reset_index(drop=True)
        return profile
    except Exception as e:
        st.error(f"Error al preparar el perfil del cliente objetivo: {e}")
        return pd.DataFrame()  # Retornar un DataFrame vacío en caso de error

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

# Inicialización de variables importantes
client_profile = pd.DataFrame()
metrics = pd.DataFrame()

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
        # Asegurar que todas las características necesarias están presentes en el DataFrame
        missing_features = [col for col in feature_names if col not in df.columns]
        for col in missing_features:
            df[col] = 0  # Agregar columnas faltantes con valores predeterminados

        # Seleccionar solo las columnas necesarias para los modelos
        df = df[feature_names]

        st.write(f"Características alineadas con los modelos: {df.columns.tolist()}")

    # Dividir datos en X e y
    if "Sexo" in df.columns:
        y = df["Sexo"]
        X = df  # No se elimina `Sexo` porque puede ser una característica usada
    else:
        y = pd.Series()
        X = df

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Calcular métricas
    metrics = calculate_metrics(models, X_test, y_test)

    # Preparar perfil del cliente objetivo
    client_profile = prepare_client_profile(df)

# Título de la aplicación
st.title("📊 Resultados del Entrenamiento y Análisis del Cliente Objetivo")

# Sección 1: Métricas de los modelos
if not metrics.empty:
    st.header("📈 Métricas de los Modelos Entrenados")
    st.dataframe(metrics)

    if st.checkbox("Mostrar gráfico de rendimiento de modelos"):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=metrics, x="Model", y="Accuracy", ax=ax, palette="viridis")
        ax.set_title("Comparación de Precisión entre Modelos")
        ax.set_ylabel("Precisión")
        ax.set_xlabel("Modelo")
        st.pyplot(fig)

# Sección 2: Cliente objetivo
if not client_profile.empty:
    st.header("👤 Cliente Objetivo")
    st.write("Características promedio del cliente objetivo:")
    st.dataframe(client_profile)

    if st.checkbox("Mostrar gráfico de distribución de características", key="dist_chart"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=client_profile, x="Value", y="Feature", ax=ax, palette="coolwarm")
        ax.set_title("Distribución de Características del Cliente Objetivo")
        ax.set_xlabel("Valor Promedio")
        st.pyplot(fig)
else:
    st.warning("El perfil del cliente objetivo está vacío o no se pudo generar.")

# Gráfico de histograma para una característica clave
if not df.empty:
    if st.checkbox("Mostrar histograma de una característica clave", key="hist_chart"):
        feature = st.selectbox("Selecciona una característica para el histograma", df.columns, key="feature_select_hist")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[feature], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Histograma de {feature}")
        ax.set_xlabel(feature)
        st.pyplot(fig)

# Gráfico de dispersión para dos características
if not df.empty:
    if st.checkbox("Mostrar gráfico de dispersión entre dos características", key="scatter_chart"):
        x_feature = st.selectbox("Selecciona la característica del eje X", df.columns, key="x_feature_scatter")
        y_feature = st.selectbox("Selecciona la característica del eje Y", df.columns, key="y_feature_scatter")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_feature, y=y_feature, ax=ax, color="green", alpha=0.7)
        ax.set_title(f"Dispersión entre {x_feature} y {y_feature}")
        st.pyplot(fig)
