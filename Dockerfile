FROM ghcr.io/huggingface/playground:latest
USER root
RUN apt-get update && apt-get install -y default-jdk wget unzip && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/joernio/joern/releases/download/v4.0.562/joern-install.sh -O /tmp/joern-install.sh && chmod +x /tmp/joern-install.sh && mkdir -p /opt/joern && /tmp/joern-install.sh --install-dir=/opt/joern && rm /tmp/joern-install.sh
ENV PATH="/opt/joern/bin:${PATH}"
USER user
