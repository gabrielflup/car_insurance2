import os
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

app = Flask(__name__)

# Função para baixar o modelo do Google Drive
def download_model_from_drive(file_id, destination):
    url = f"https://drive.google.com/file/d/11H8P8NhajaYk70-OtAfguLc0oQXv_ZV_/view?usp=sharing"
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(destination, "wb") as f:
            f.write(response.content)
        print(f"Modelo salvo como {destination}")
    else:
        print("Erro ao baixar o modelo!")
        raise Exception("Falha no download do modelo.")

# ID do arquivo no Google Drive (substitua pelo seu)
FILE_ID = "1ABCDEFGH"  # Substitua pelo ID do seu arquivo
MODEL_PATH = "classify_model (1).h5"

# Baixar o modelo se ele não existir localmente
if not os.path.exists(MODEL_PATH):
    print("Baixando o modelo...")
    download_model_from_drive(FILE_ID, MODEL_PATH)

# Carregar o modelo
print("Carregando o modelo...")
model = load_model(MODEL_PATH)
print("Modelo carregado com sucesso!")

# Rota principal para predições
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Verificar se o arquivo foi enviado
        if 'image' not in request.files:
            return jsonify({"error": "Nenhuma imagem enviada"}), 400
        
        # Ler a imagem enviada
        image = request.files["image"]
        image = Image.open(image).convert("RGB")  # Converte para RGB caso seja diferente

        # Redimensionar a imagem para o tamanho esperado pelo modelo (ajuste conforme necessário)
        image = image.resize((224, 224))  # Substitua (224, 224) pelo tamanho correto do modelo
        image = img_to_array(image) / 255.0  # Normalizar os valores para [0, 1]
        image = np.expand_dims(image, axis=0)  # Adicionar uma dimensão para batch

        # Realizar a predição
        predictions = model.predict(image)
        result = predictions.tolist()  # Converter para lista para enviar como JSON

        return jsonify({"predictions": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Rota de saúde (para verificar se a API está funcionando)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
