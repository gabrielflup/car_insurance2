# Usar uma imagem base do Python (versão slim para ser mais leve)
FROM python:3.8-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos do projeto para o container
COPY . /app

# Instalar as dependências do arquivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta (não é necessário fixar a porta, o Render vai fornecer)
EXPOSE 5000  # Esta linha pode ser mantida, já que você está configurando a porta dinamicamente via variável de ambiente

# Comando para rodar a aplicação FastAPI com Uvicorn, usando a variável de ambiente PORT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]  # Aqui você pode substituir 5000 por 8000 ou até a variável de ambiente, conforme necessário
