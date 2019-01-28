# Serving Tensorflow Estimator with Tensorflow Serving Examples

In this repository, I would like to show examples to server tensorflow estimator with tensorflow serving.
The goals of this repository are to understand:
1. how to create tensorflow estimators to serve with tensorflow serving,
2. how to server a model with tensorflow, and
3. how to implement a gRPC client for tensorflow serving.

At the time when I am writing this, there are less documentation about serving custom tensorflow estimator.
That's why I made this for myself.
I hope it would help someone with learning something similar.

## Requirements
- Python 2.7
- Docker
- Anaconda

## How to run the examples

### 1. You clone the repository.
```
git clone git@github.com:yu-iskw/tensorflow-serving-example.git
```

### 2. You create an anaconda environment.
At the moment when I am creating this, versions of main components are the following.
- tensorflow: 1.12.0
- tensorflow-serving-api: 1.12.0
- tensorflow-serving: 1.12.0
```
conda env create -f environment.yml
```

Once you create the environment, you need to activate this using the following command:

```
source activate tensorflow-serving-example
```

### 3. You train a model.
You train a model with `python/train/mnist_custom_estimator.py`.
It is used for training a model to the MNIST task.
```
python python/train/mnist_custom_estimator.py \
    --steps 100 \
    --saved_dir ./models/ \
    --model_dir /tmp/mnist_custom_estimator
```

Where `--steps` is the number of steps to train a model, `--saved_dir` is a path to save the model for tensorflow serving  and `--model_dir` is a path to save traditional checkpoints, meta graph and so on.

When you finish the trained model, the saved model exists under `./models` directory like below.
Here, `1548714304` is the unix timestamp when the model was saved.
This is different from what you get.
It depends on when you run the python code.

We can server `saved_model.pb` with tensorflow serving.
```
./models
└── 1548714304
    ├── saved_model.pb
        └── variables
                ├── variables.data-00000-of-00001
                        └── variables.index
```

### 4. You build a docker image for tensorflow serving.
As you can see `./Dockerfile`, it just installs the pre-built tensorflow serving package.
This is because building it takes a little long time to compile tensorflow serving.
If you want to make a docker image from the tensorflow serving source, [those docker files](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/tools/docker) would help.
```
docker build --rm -f Dockerfile -t tensorflow-serving-example:0.6 .
```

At the time when I am writing this, there is something wrong with `tensorflow-serving-universal`.
If you are interested in the issue, please track [Package recently broken on ubuntu 16\.04 ? · Issue \#819 · tensorflow/serving](https://github.com/tensorflow/serving/issues/819).


### 5. You run the docker container with the trained model.
Before running a docker container, you must prepare for the served model.
```
# Prepare for the served model.
mkdir -p ./models_for_serving/mnist/1
cp -R ./models/mnist_custom_estimator/pb/1548714304/* ./models_for_serving/mnist/1

# As a result of copying the files, the directory should be like following.
models_for_serving/
└── mnist
    └── 1
        ├── saved_model.pb
        └── variables
            ├── variables.data-00000-of-00001
            └── variables.index
```
As you probably know, tenwoflow can handle multiple versions of served models.
`1` at the tail of `./models_for_serving/mnist/1/` means the served model version.

We have prepared for the served model.
Now let's move on to running a docker container to server the model.
The model serving supports not only gRPC API, but HTTP/REST API.
The port 8500 is used for gRPC.
Meanwhile, the port 8501 is used for RESTful API.
```
# Run a docker container.
docker run --rm -it -v /Users/yuishikawa/local/src/github/tensorflow-serving-example/models_for_serving:/models \
    -e MODEL_NAME=mnist \
    -e MODEL_PATH=/models/mnist \
    -p 8500:8500  \
    -p 8501:8501  \
    --name tensorflow-serving-example \
    tensorflow-serving-example:0.6
```

Where `MODEL_NAME` is an environment variable to identify the served model when a gRPC client requests,
`MODEL_PATH` is an environment variable to identify the path to the saved model.
Besides, we share `./models_for_serving` between docker host and guest.

### 6. You create a gRPC client for tensorflow serving.
`./python/grpc_mnist_client.py` is an example of gRPC client for tensorflow serving.
```
TENSORFLOW_SERVER_HOST="..."
python python/grpc_mnist_client.py \
  --image ./data/0.png \
  --model mnist \
  --host $TENSORFLOW_SERVER_HOST
```
You can pass one with `--image` option.
I already prepared for some sample MNIST images in `./data/`.
If you use `docker-machine`, you can get the host with `docker-machine ip`.
`--model mnist` is defined, when running a docker container with `-e MODEL_NAME='mnist'`.

## Appendix: Serving premodeled tensorflow estimator
`./python/train/mnist_premodeled_estimator.py` is an example to save a trained model with a premodeled tensorflow estimator.
One of the differences from a custom tensorflow estimator is the model spec signature name.
When saving a model with a custom tensorflow estimator, the signature name is `serving_default` by default.
On the other hand, when saving a model with a pre-modeled tensorflow estimator, the signature name is `predict` by default.
```
TENSORFLOW_SERVER_HOST="..."
python python/grpc_mnist_client.py \
  --image ./data/0.png \
  --model mnist \
  --host $TENSORFLOW_SERVER_HOST \
  --signature_name predict
```
