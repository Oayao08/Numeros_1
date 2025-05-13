from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Configuración de Flask
app = Flask(__name__)

# Cargar el modelo
model = tf.keras.models.model_from_json(open('model_numeros.json').read())
model.load_weights('model_numeros.weights.h5')

# Directorio de las imágenes subidas
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Función para procesar la imagen subida
def prepare_image(image):
    image = Image.open(image)
    image = image.resize((100, 100))
    image = np.array(image) / 255.0  # Normalizar la imagen
    image = np.expand_dims(image, axis=0)  # Agregar una dimensión para el batch
    return image

# Página principal
@app.route('/')
def index():
    return render_template('index.html')

# Página para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocesar la imagen
        image = prepare_image(filepath)
        
        # Realizar la predicción
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
