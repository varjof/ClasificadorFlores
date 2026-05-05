
import streamlit as st
import numpy as np
import joblib
import os

# Obtener el directorio actual donde se está ejecutando el script de Streamlit
# Esto es importante si los archivos .joblib están en el mismo directorio
script_dir = os.path.dirname(__file__)

# Construir las rutas completas a los archivos .joblib
model_path = os.path.join(script_dir, 'mlp_model.joblib')
label_encoder_path = os.path.join(script_dir, 'label_encoder (1).joblib')

# Cargar el modelo y el LabelEncoder
# Asegúrate de que estos archivos estén disponibles en el mismo directorio que tu script de Streamlit
@st.cache_resource
def load_resources():
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, label_encoder

model, label_encoder = load_resources()

st.title('Predicción de Especie de Flor')
st.write('Introduce las características de la flor para predecir su especie.')

# Cuadros de entrada para las características de la flor
sepal_length = st.number_input('Largo del Sépalo (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Ancho del Sépalo (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Largo del Pétalo (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Ancho del Pétalo (cm)', min_value=0.0, max_value=10.0, value=1.5)

if st.button('Predecir'):
    # Crear un array numpy con las características de entrada
    # Asegúrate de que el orden de las características coincida con el esperado por el modelo
    flower_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Realizar la predicción
    prediction = model.predict(flower_features)

    # Como vimos, el modelo ya predice la etiqueta de texto directamente
    predicted_species = prediction[0]

    st.success(f'La especie predicha para la flor es: **{predicted_species}**')

st.warning('Nota: Este modelo fue entrenado con datos de Iris. Los resultados para otras flores pueden variar.')
