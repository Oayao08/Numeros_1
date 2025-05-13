import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

# Cargar el modelo completo desde archivo .keras
model = load_model('modelo.keras')


# Función para hacer predicción
def predict_image(image):
    image = image.resize((100, 100))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions

# Configuración de la interfaz
st.title('Clasificador de Números (1-11)')
st.write("Cargar una imagen del número del 1 al 11.")

uploaded_image = st.file_uploader("Cargar Imagen", type=["jpg", "png"])

if uploaded_image is not None:
    from PIL import Image
    image = Image.open(uploaded_image)
    st.image(image, caption='Imagen cargada', use_column_width=True)

    # Realizar predicción
    predictions = predict_image(image)
    predicted_class = predictions.argmax()  # Clase predicha

    st.write(f"Predicción: Número {predicted_class + 1}")
