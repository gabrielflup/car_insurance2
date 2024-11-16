from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

app = Flask(__name__)

# URL do arquivo no Google Drive
MODEL_URL = "https://drive.google.com/uc?id=11H8P8NhajaYk70-OtAfguLc0oQXv_ZV_"
MODEL_PATH = "classify_model.h5"

# Função para baixar o modelo
def download_model():
    if not os.path.exists(MODEL_PATH):  # Baixa o modelo apenas se não estiver no diretório
        print("Baixando o modelo do Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Download concluído.")

# Baixe o modelo
download_model()

# Carregar o modelo
print("Carregando o modelo...")
model = load_model(MODEL_PATH)
print("Modelo carregado com sucesso.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verificar se o arquivo foi enviado
        if 'file' not in request.files:
            return jsonify({"error": "Nenhuma imagem enviada."}), 400
        
        file = request.files['file']

        # Abrir a imagem
        img = Image.open(file).resize((224, 224))  # Certifique-se de usar o tamanho esperado pelo modelo
        img_array = np.array(img) / 255.0  # Normalização
        img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão batch

        # Fazer a predição
        prediction = model.predict(img_array)
        result = prediction.tolist()

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
