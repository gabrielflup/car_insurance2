FROM python:3.9-slim

# Defina o diretório de trabalho
WORKDIR /app

# Copie o arquivo requirements.txt
COPY requirements.txt /app/requirements.txt

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie os arquivos do projeto
COPY . /app


EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
