import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from PIL import Image
import numpy as np
import cv2
import gdown

# Caminho para o modelo
MODEL_URL = "https://drive.google.com/uc?id=11H8P8NhajaYk70-OtAfguLc0oQXv_ZV_"
MODEL_PATH = "classify_model.h5"

# Função para baixar o modelo
def download_model():
    if not os.path.exists(MODEL_PATH):  # Baixa o modelo apenas se não estiver no diretório
        print("Baixando o modelo do Google Drive...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            print("Download concluído.")
        except Exception as e:
            print(f"Erro ao baixar o modelo: {str(e)}")
            raise

# Função para carregar o modelo com a camada personalizada
def load_custom_model():
    try:
        with custom_object_scope({"CustomScaleLayer": CustomScaleLayer}):  # Registrar camada personalizada
            model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {str(e)}")
        raise

# Baixar o modelo
download_model()

# Carregar o modelo com a camada personalizada registrada
model = load_custom_model()

# Iniciar a aplicação Flask
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Nenhuma imagem enviada"}), 400
        
        # Ler a imagem enviada
        image = request.files["image"]
        image = Image.open(image).convert("RGB")  # Converte para RGB
        image = image.resize((299, 299))  # Ajuste de tamanho conforme esperado pelo modelo
        
        # Pré-processamento da imagem
        image = np.array(image)  # Converter a imagem em um array numpy
        image = image / 255.0  # Normalizar os valores dos pixels para o intervalo [0, 1]
        image = np.expand_dims(image, axis=0)  # Adicionar uma dimensão extra para corresponder ao formato de entrada do modelo

        # Fazer a previsão
        prediction = model.predict(image)
        
        # Interpretação do resultado
        result = "Carro Em Bom Estado" if prediction > 0.2 else "Carro Batido"
        
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
