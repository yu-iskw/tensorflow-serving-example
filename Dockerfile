FROM ubuntu:18.04

#
# Reference: https://github.com/tensorflow/serving/issues/819
#

# Install general packages
RUN apt-get update && apt-get install -y \
        curl \
        libcurl3-dev \
        unzip \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Previous Installation of tensorflow-model-server (BROKEN RECENTLY)
#RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list \
#    && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - \
#    && apt-get update && apt-get install tensorflow-model-server

# New installation of tensorflow-model-server
RUN TEMP_DEB="$(mktemp)" \
    && wget -O "$TEMP_DEB" 'http://storage.googleapis.com/tensorflow-serving-apt/pool/tensorflow-model-server-1.12.0/t/tensorflow-model-server/tensorflow-model-server_1.12.0_all.deb' \
    && dpkg -i "$TEMP_DEB" \
    && rm -f "$TEMP_DEB"

# gRPC port
EXPOSE 8500
# REST API port
EXPOSE 8501

# Serve the model when the container starts
CMD tensorflow_model_server \
  --port=8500 \
  --rest_api_port=8501 \
  --model_name="$MODEL_NAME" \
  --model_base_path="$MODEL_PATH"
