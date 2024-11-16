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

# Função para baixar o modelo do Google Drive
def download_model():
    if not os.path.exists(MODEL_PATH):  # Baixa o modelo apenas se não estiver no diretório
        print("Baixando o modelo do Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Download concluído.")

# Baixar o modelo
download_model()

# Carregar o modelo
print("Carregando o modelo...")
model = load_model(MODEL_PATH)
print("Modelo carregado com sucesso.")

# Rota de predição
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Nenhuma imagem enviada"}), 400
        
        # Ler a imagem enviada
        image = request.files["image"]
        image = Image.open(image).convert("RGB")  # Converte para RGB
        image = image.resize((224, 224))  # Ajuste de tamanho conforme esperado pelo modelo
        image = np.array(image) / 255.0  # Normalizar a imagem
        image = np.expand_dims(image, axis=0)  # Adiciona a dimensão de batch

        # Realizar a predição
        predictions = model.predict(image)
        result = predictions.tolist()  # Converte para lista para enviar no JSON

        return jsonify({"predictions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Rota de saúde
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
