FROM python:3.12.9-slim

WORKDIR /app

COPY . /app/

RUN pip install -r requirements.txt

RUN pip uninstall -y pinecone-client pinecone

RUN pip install --upgrade pinecone

RUN pip install --upgrade langchain_pinecone

CMD [ "uvicorn","app:app","--host","0.0.0.0","--port","5000"]