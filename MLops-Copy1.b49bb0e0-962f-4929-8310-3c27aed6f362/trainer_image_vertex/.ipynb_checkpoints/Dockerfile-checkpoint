FROM us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest
RUN pip install fire cloudml-hypertune pandas scikit-learn==1.2.2

# WORKDIR /app
WORKDIR /app

COPY train.py .
ENTRYPOINT ["python", "train.py"]