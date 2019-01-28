NAME := tensorflow-serving-example
VERSION := 0.6

.PHONY: build-docker run-docker create-conda remove-conda test

build-docker:
	docker build --rm -f Dockerfile -t $(NAME):$(VERSION) .
	docker build --rm -f Dockerfile -t $(NAME):dev .

run-docker: check-model-name
	docker run --rm -v ${PWD}/models_for_serving:/models \
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
	rm -fr ./models/mnist_keras_estimator/
	mkdir -p ./models/mnist_keras_estimator/
	python python/train/mnist_keras_estimator.py \
			--max_steps 1000 \
			--model_dir ./models/mnist_keras_estimator/

train-iris-premodeled-estimator:
	rm -fr ./models/iris_premodeled_estimator/
	python python/train/iris_premodeled_estimator.py \
			--max_steps 10000 \
			--model_dir ./models/iris_premodeled_estimator/

run-mnist-grpc-client:
	python python/grpc_mnist_client.py \
		--image ./data/0.png \
		--model mnist \
		--host "$(shell docker-machine ip default)"

run-iris-grpc-client:
	python python/grpc_iris_client.py \
		--model iris \
		--host "$(shell docker-machine ip default)" \
		--signature_name "predict"

check-model-name:
ifndef MODEL_NAME
	$(error MODEL_NAME is not set)
endif
