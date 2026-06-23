FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk-headless \
    curl \
    wget \
    unzip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/joernio/joern/releases/download/v4.0.562/joern-install.sh -O /tmp/joern-install.sh && \
    chmod +x /tmp/joern-install.sh && \
    mkdir -p /opt/joern && \
    /tmp/joern-install.sh --install-dir=/opt/joern && \
    rm /tmp/joern-install.sh

ENV PATH="/opt/joern/joern-cli:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
