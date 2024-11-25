from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from PIL import Image
import numpy as np
import os
import requests
import tensorflow as tf
import uvicorn

# Função para baixar o modelo do Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = f"https://drive.google.com/uc?id={file_id}"
    #URL = "https://drive.google.com/file/d/11H8P8NhajaYk70-OtAfguLc0oQXv_ZV_/view?usp=sharing"
    response = requests.get(URL, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(f"Modelo baixado para: {destination}")

# Função para processar a imagem
def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((299, 299))  # O InceptionResNetV2 requer imagens 299x299
    image = np.array(image) / 299.0  # Normaliza a imagem
    image = np.expand_dims(image, axis=0)  # Expande para o formato batch
    return image

# ID do arquivo no Google Drive
MODEL_ID = "11H8P8NhajaYk70-OtAfguLc0oQXv_ZV_"  # ID do arquivo do Google Drive
MODEL_PATH = "classify_model.h5"  # Caminho local para salvar o modelo

# Baixar o modelo ao iniciar
download_file_from_google_drive(MODEL_ID, MODEL_PATH)

# Criar o modelo com o InceptionResNetV2
base_model = InceptionResNetV2(include_top=False, input_shape=(299, 299, 3))
classification_model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Usando uma saída binária
])

# Carregar os pesos do modelo
classification_model.load_weights(MODEL_PATH)

# Configuração do FastAPI
app = FastAPI()

# Middleware para permitir acesso
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Endpoint raiz com HTML para upload
@app.get("/", response_class=HTMLResponse)
async def main():
    content = """
    <html>
        <head>
            <title>Car Status Predictor</title>
        </head>
        <body>
            <h1>Upload uma Imagem do Carro</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <input type="submit" value="Prever">
            </form>
        </body>
    </html>
    """
    return content

# Endpoint para predição
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Salvar o arquivo temporariamente
        temp_file_path = "temp_image.jpg"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Processar e prever
        image = process_image(temp_file_path)
        prediction = classification_model.predict(image)

        # Limpar o arquivo temporário
        os.remove(temp_file_path)

        # Resultado da predição
        result = "Carro em bom estado" if prediction > 0.2 else "Carro batido"
        return HTMLResponse(content=f"<h1>Predição: {result}</h1><p>Valor de confiança: {prediction[0][0]:.2f}</p>", status_code=200)

    except Exception as e:
        return {"error": str(e)}

# Rodar o servidor
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Se a variável de ambiente PORT não estiver definida, use a porta 5000
    uvicorn.run(app, host="0.0.0.0", port=port)
