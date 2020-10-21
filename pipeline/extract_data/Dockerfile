FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy pandas

COPY extract_data.py ./extract_data.py

COPY ../../params.yaml  ./params.yaml

ENTRYPOINT [ "python", "extract_data.py" ]