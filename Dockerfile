#Dockerfile
FROM python:3.8
WORKDIR /app
COPY ./requeriments.txt ./requeriments.txt
RUN pip install --no-cache-dir --upgrade -r ./requeriments.txt
COPY ./app.py .
COPY ./model ./model
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "31"]