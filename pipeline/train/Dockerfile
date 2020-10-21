FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy pandas

COPY train.py ./train.py

ENTRYPOINT [ "python", "train.py" ]