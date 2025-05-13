import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Intentar cargar el modelo
try:
    model = load_model('modelo.keras')
    st.success("Modelo cargado con éxito")
except Exception as e:
    st.error(f"No se pudo cargar el modelo. Error: {str(e)}")

# Función para hacer predicción
def predict_image(image):
    # Asegúrate de que la imagen tiene el tamaño correcto y la forma adecuada para la red
    image = image.convert('RGB')  # Convertir imagen a RGB si no está en ese formato
    image = image.resize((100, 100))  # Asegúrate de que la imagen tiene el tamaño correcto
    image = np.array(image).astype('float32') / 255.0  # Normalizar la imagen (a float32)
    image = np.expand_dims(image, axis=0)  # Añadir la dimensión del batch
    
    # Verifica si el modelo espera una imagen en escala de grises o color (100, 100, 1) o (100, 100, 3)
    if image.shape[-1] == 3:  # Si tiene 3 canales (RGB)
        predictions = model.predict(image)  # Predicción del modelo
    else:
        st.error("El modelo no está configurado para imágenes con esta cantidad de canales")
        return None
    
    return predictions

# Configuración de la interfaz
st.title('Clasificador de Números (1-11)')
st.write("Cargar una imagen del número del 1 al 11.")

uploaded_image = st.file_uploader("Cargar Imagen", type=["jpg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)  # Abrir la imagen cargada
    st.image(image, caption='Imagen cargada', use_column_width=True)

    # Realizar predicción
    predictions = predict_image(image)
    if predictions is not None:
        predicted_class = predictions.argmax()  # Clase predicha
        st.write(f"Predicción: Número {predicted_class + 1}")
