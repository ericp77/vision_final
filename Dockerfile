FROM nvcr.io/nvidia/pytorch:23.03-py3
COPY requirements.txt /workspace
#COPY dataset /workspace
#COPY src /workspace
WORKDIR /workspace
RUN pip install --no-cache-dir -r requirements.txt