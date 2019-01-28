from __future__ import print_function

import argparse
import time

import grpc
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def run(host, port, model, signature_name):

    channel = grpc.insecure_channel('{host}:{port}'.format(host=host, port=port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    start = time.time()

    # Call classification model to make prediction
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    request.inputs['sepal_length'].CopyFrom(make_tensor_proto(6.8, shape=[1, 1]))
    request.inputs['sepal_width'].CopyFrom(make_tensor_proto(3.2, shape=[1, 1]))
    request.inputs['petal_length'].CopyFrom(make_tensor_proto(5.9, shape=[1, 1]))
    request.inputs['petal_width'].CopyFrom(make_tensor_proto(2.3, shape=[1, 1]))

    result = stub.Predict(request, 10.0)

    end = time.time()
    time_diff = end - start

    # Reference:
    # How to access nested values
    # https://stackoverflow.com/questions/44785847/how-to-retrieve-float-val-from-a-predictresponse-object
    print(result)
    print('time elapased: {}'.format(time_diff))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Tensorflow server host name', default='localhost', type=str)
    parser.add_argument('--port', help='Tensorflow server port number', default=8500, type=int)
    parser.add_argument('--model', help='model name', type=str)
    parser.add_argument('--signature_name', help='Signature name of saved TF model',
                        default='serving_default', type=str)

    args = parser.parse_args()
    run(args.host, args.port, args.model, args.signature_name)
