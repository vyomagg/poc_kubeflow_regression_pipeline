FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy pandas

COPY prepare.py ./prepare.py

COPY ../../params.yaml  ./params.yaml

ENTRYPOINT [ "python", "prepare.py" ]