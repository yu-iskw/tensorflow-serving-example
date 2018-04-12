#!/bin/bash -e

# Generate files from tensorflow proto
python -m grpc.tools.protoc \
  -I=tensorflow/ \
  --python_out=./python/ \
  --proto_path=. \
  ./tensorflow/tensorflow/core/framework/*.proto
python -m grpc.tools.protoc \
  -I=tensorflow/ \
  --python_out=./python/ \
  --proto_path=. \
  ./tensorflow/tensorflow/core/example/*.proto
python -m grpc.tools.protoc \
  -I=tensorflow/ \
  --python_out=./python/ \
  --proto_path=. \
  ./tensorflow/tensorflow/core/protobuf/*.proto

# Generate files from tensorflow serving proto
python -m grpc.tools.protoc \
  -I=tensorflow/ \
  -I=serving/ \
  --python_out=./python/ \
  --proto_path=. \
  serving/tensorflow_serving/apis/*.proto
