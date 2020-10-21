FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy pandas

COPY deploy.py ./deploy.py

ENTRYPOINT [ "python", "evaluate.py" ]