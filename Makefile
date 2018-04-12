NAME := tensorflow-serving-example
VERSION := 0.1

.PHONY: build-docker run-docker create-conda remove-conda test

build-docker:
	docker build --rm -f Dockerfile -t $(NAME):$(VERSION) .
	docker build --rm -f Dockerfile -t $(NAME):dev .

run-docker:
	docker run --rm  -v ${PWD}/models_for_serving:/models \
		-e MODEL_NAME='mnist' \
		-e MODEL_PATH='/models/mnist_custom_estimator' \
		-p 8500:8500  \
		--name $(NAME) \
		$(NAME):$(VERSION)

create-conda:
	conda env create -f environment.yml -n $(NAME)

remove-conda:
	conda env remove -y -n $(NAME)
