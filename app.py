from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image

# Carregar o modelo
best_class_model = keras.models.load_model('/classify_model (1).h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo encontrado'}), 400
    
    file = request.files['file']
    
    # Processar a imagem recebida
    image = Image.open(file)
    image = image.resize((299, 299))  # Redimensionar a imagem
    image = np.array(image) / 255.0  # Normalizar
    image = np.expand_dims(image, axis=0)  # Adicionar dimensão extra para o modelo

    # Fazer a previsão
    prediction = best_class_model.predict(image)
    
    # Resultado da previsão
    result = 'Carro Em Bom Estado' if prediction > 0.2 else 'Carro Batido'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
