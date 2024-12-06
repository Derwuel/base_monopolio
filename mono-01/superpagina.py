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

# Estilos CSS personalizados
custom_css = """
<style>
    /* General styling */
    body {
        background-color: #f9f9f9;
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        text-align: center;
        color: #4CAF50;
        font-size: 36px;
        margin-bottom: 20px;
    }
    .sub-title {
        color: #4CAF50;
        font-size: 24px;
        margin-bottom: 10px;
    }

    /* Layout adjustments */
    .stColumn > div {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        height: auto !important;
    }

    /* Data table styling */
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Chart area */
    .chart-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 10px;
    }

    /* Alert message styling */
    .stAlert {
        background-color: #e7f3e7;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
    }

    /* Radio buttons */
    .stRadio > div {
        display: flex;
        justify-content: space-evenly;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #888;
    }
</style>
"""

# Incluir estilos personalizados en la página
st.markdown(custom_css, unsafe_allow_html=True)

# Título principal
st.markdown("<div class='main-title'>Análisis de Datos</div>", unsafe_allow_html=True)

# Dividir el diseño en dos columnas
col1, col2 = st.columns([1, 2], gap="large")

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
            if hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
    return models, feature_names

# Función para calcular métricas de los modelos
def calculate_metrics(models, X_test, y_test):
    """Calcular métricas para cada modelo."""
    metrics_list = []
    for model in models:
        try:
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
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
        return pd.DataFrame()

# Leer datos desde un archivo (modifica la ruta si es necesario)
data_file = "data/02_intermediate/pos_proceso.pq"
try:
    df = pd.read_parquet(data_file)
    file_loaded_message = "Archivo Parquet cargado correctamente."
except Exception as e:
    df = pd.DataFrame()
    file_loaded_message = f"Error al cargar el archivo Parquet: {e}"

# Preparar métricas y cliente objetivo si el DataFrame no está vacío
if not df.empty:
    pickle_files = [
        "data/06_models/tree_model.pickle",
        "data/06_models/resultado01.pickle",
        "data/06_models/svr_model.pickle",
    ]
    models, feature_names = load_models(pickle_files)

    if feature_names is not None:
        missing_features = [col for col in feature_names if col not in df.columns]
        for col in missing_features:
            df[col] = 0
        df = df[feature_names]

    if "Sexo" in df.columns:
        y = df["Sexo"]
        X = df
    else:
        y = pd.Series()
        X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    metrics = calculate_metrics(models, X_test, y_test)
    client_profile = prepare_client_profile(df)
else:
    metrics = pd.DataFrame()
    client_profile = pd.DataFrame()

# Cliente objetivo (columna izquierda)
with col1:
    st.markdown("<div class='sub-title'>Cliente Objetivo</div>", unsafe_allow_html=True)
    if not client_profile.empty:
        st.dataframe(client_profile, use_container_width=True)
    else:
        st.warning("El perfil del cliente objetivo está vacío o no se pudo generar.")

# Visualización de gráficos (columna derecha)
with col2:
    st.markdown("<div class='sub-title'>Gráficos</div>", unsafe_allow_html=True)

    # Botones de selección para gráficos
    selected_graph = st.radio(
        "Selecciona el tipo de gráfico:",
        options=["Gráfico de distribución", "Histograma", "Gráfico de dispersión"],
        horizontal=True
    )

    plot_area = st.empty()

    if selected_graph == "Gráfico de distribución" and not client_profile.empty:
        fig, ax = plt.subplots(figsize=(5, 4))  # Tamaño moderado del gráfico
        sns.barplot(data=client_profile, x="Value", y="Feature", palette="coolwarm", ax=ax)
        ax.set_title("Distribución de Características")
        plot_area.pyplot(fig)

    elif selected_graph == "Histograma" and not df.empty:
        feature = st.selectbox("Selecciona una característica para el histograma", df.columns)
        fig, ax = plt.subplots(figsize=(5, 4))  # Tamaño moderado del gráfico
        sns.histplot(df[feature], kde=True, color="skyblue", ax=ax)
        ax.set_title(f"Histograma de {feature}")
        plot_area.pyplot(fig)

    elif selected_graph == "Gráfico de dispersión" and not df.empty:
        x_feature = st.selectbox("Eje X", df.columns)
        y_feature = st.selectbox("Eje Y", df.columns)
        fig, ax = plt.subplots(figsize=(5, 4))  # Tamaño moderado del gráfico
        sns.scatterplot(data=df, x=x_feature, y=y_feature, color="green", alpha=0.7, ax=ax)
        ax.set_title(f"Dispersión entre {x_feature} y {y_feature}")
        plot_area.pyplot(fig)
    else:
        st.warning("No hay datos disponibles para mostrar este gráfico.")

# Mostrar mensaje de carga del archivo después de los gráficos
st.markdown(f"<div class='stAlert'>{file_loaded_message}</div>", unsafe_allow_html=True)

# Pie de página
st.markdown("<div class='footer'>© 2024 - Análisis de Datos</div>", unsafe_allow_html=True)
