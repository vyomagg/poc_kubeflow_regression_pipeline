FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy pandas

COPY evaluate.py ./evaluate.py

ENTRYPOINT [ "python", "evaluate.py" ]