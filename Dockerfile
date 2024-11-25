# Usar uma imagem base do Python (versão slim para ser mais leve)
FROM python:3.8-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos do projeto para o container
COPY . /app

# Instalar as dependências do arquivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta 5000 para o Flask
EXPOSE 5000

# Comando para rodar a aplicação FastAPI com Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
