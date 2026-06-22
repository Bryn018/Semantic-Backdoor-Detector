FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    wget \
    unzip \
    jq \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install Joern CLI v4.0.562 (~2GB download)
RUN mkdir -p /opt/joern && \
    cd /tmp && \
    wget -q "https://github.com/joernio/joern/releases/download/v4.0.562/joern-cli.zip" -O joern-cli.zip && \
    unzip -q joern-cli.zip -d /opt/joern && \
    chmod -R +x /opt/joern/ && \
    rm -f joern-cli.zip

ENV JOERN_HOME=/opt/joern
ENV PATH="/opt/joern/bin:${PATH}"

# Copy application code
WORKDIR /app
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set entrypoint
ENTRYPOINT ["bash", "github_action/entrypoint.sh"]
