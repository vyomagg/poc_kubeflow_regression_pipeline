FROM python:3.7-slim

WORKDIR /app

RUN pip install -U scikit-learn numpy pandas

COPY prepare.py ./prepare.py

ENTRYPOINT [ "python", "prepare.py" ]