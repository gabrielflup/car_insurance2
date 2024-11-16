from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from PIL import Image
import numpy as np
import os
import gdown

# Inicializar o Flask
app = Flask(__name__)

# Caminho para o modelo e configuração
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

# Configuração do modelo
print("Carregando o modelo...")
base_model = InceptionResNetV2(include_top=False, input_shape=(299, 299, 3))
classification_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Carregar os pesos
classification_model.load_weights(MODEL_PATH)

# Função para pré-processar imagens
def preprocess_image(image):
    image = image.resize((299, 299))  # Redimensionar para o tamanho esperado pelo modelo
    image = np.array(image) / 255.0  # Normalizar os valores dos pixels
    return np.expand_dims(image, axis=0)

# Rota para previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verificar se o arquivo foi enviado
        if 'file' not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        file = request.files['file']

        # Abrir e pré-processar a imagem
        image = Image.open(file)
        image = preprocess_image(image)

        # Fazer previsão
        prediction = classification_model.predict(image)[0][0]

        # Decisão com base na predição
        result = "Carro Em Bom Estado" if prediction > 0.2 else "Carro Batido"
        return jsonify({"prediction": float(prediction), "result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Rota de saúde (opcional, útil para monitoramento no Render)
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# Rodar o aplicativo no Render
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
