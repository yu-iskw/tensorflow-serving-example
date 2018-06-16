NAME := tensorflow-serving-example
VERSION := 0.3

.PHONY: build-docker run-docker create-conda remove-conda test

MODEL_PATH := /models/mnist_custom_estimator

build-docker:
	docker build --rm -f Dockerfile -t $(NAME):$(VERSION) .
	docker build --rm -f Dockerfile -t $(NAME):dev .

run-docker:
	docker run --rm  -v ${PWD}/models_for_serving:/models \
		-e MODEL_NAME=$(MODEL_NAME) \
		-e MODEL_PATH=/models/$(MODEL_NAME) \
		-p 8500:8500  \
		-p 8501:8501  \
		--name $(NAME) \
		$(NAME):$(VERSION)

stop-docker:
	docker stop $(NAME)

create-conda:
	conda env create -f environment.yml -n $(NAME)

remove-conda:
	conda env remove -y -n $(NAME)

lint:
	flake8

train-custom-estimator:
	rm -fr ./models/mnist_custom_estimator/pb/
	rm -fr ./models/mnist_custom_estimator/ckpt/
	python python/train/mnist_custom_estimator.py \
			--steps 200 \
			--saved_dir ./models/mnist_custom_estimator/pb/ \
			--model_dir ./models/mnist_custom_estimator/ckpt/ \

train-premodeled-estimator:
	rm -fr ./models/mnist_premodeled_estimator/pb/
	rm -fr ./models/mnist_premodeled_estimator/ckpt/
	python python/train/mnist_premodeled_estimator.py \
			--steps 10000 \
			--saved_dir ./models/mnist_premodeled_estimator/pb/ \
			--model_dir ./models/mnist_premodeled_estimator/ckpt/ \

train-keras-estimator:
	rm -fr ./models/mnist_keras_estimator/pb/
	rm -fr ./models/mnist_keras_estimator/ckpt/
	mkdir -p ./models/mnist_keras_estimator/pb/
	mkdir -p ./models/mnist_keras_estimator/ckpt/
	python python/train/mnist_keras_estimator.py \
			--steps 1000 \
			--saved_dir ./models/mnist_keras_estimator/pb/ \
			--model_dir ./models/mnist_keras_estimator/ckpt/ \

train-iris-premodeled-estimator:
	rm -fr ./models/iris_premodeled_estimator/pb/
	rm -fr ./models/iris_premodeled_estimator/ckpt/
	python python/train/iris_premodeled_estimator.py \
			--steps 100 \
			--saved_dir ./models/iris_premodeled_estimator/pb/ \
			--model_dir ./models/iris_premodeled_estimator/ckpt/ \

run-grpc-client:
	python python/grpc_client.py \
		--image ./data/0.png \
		--model mnist \
		--host "$(docker-machine ip default)"
