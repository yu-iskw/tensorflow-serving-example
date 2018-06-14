import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import models

tf.app.flags.DEFINE_integer('steps', 100, 'The number of steps to train a model')
tf.app.flags.DEFINE_string('model_dir', '/tmp/mnist_keras_estimator', 'Dir to save a model and checkpoints')
tf.app.flags.DEFINE_string('saved_dir', './models/', 'Dir to save a model for TF serving')
FLAGS = tf.app.flags.FLAGS

# This is used to specify the input parameter.
INPUT_FEATURE = 'image'
INPUT_SHAPE = 28 * 28 * 1
NUM_CLASSES = 10


def get_keras_model():
    inputs = layers.Input(shape=(INPUT_SHAPE,), name=INPUT_FEATURE)
    dense256 = layers.Dense(256, activation='relu')(inputs)
    dense32 = layers.Dense(32, activation='relu')(dense256)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(dense32)
    model = models.Model(inputs, outputs)
    return model


def serving_input_receiver_fn():
    """
    This is used to define inputs to serve the model.

    :return: ServingInputReciever
    """
    reciever_tensors = {
        # The size of input image is flexible.
        INPUT_FEATURE: tf.placeholder(tf.float32, [None, None, None, 1]),
    }

    # Convert give inputs to adjust to the model.
    features = {
        # Resize given images.
        INPUT_FEATURE: tf.reshape(reciever_tensors[INPUT_FEATURE], [-1, INPUT_SHAPE])
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                                                    features=features)


def main(_):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    train_labels = keras.utils.to_categorical(train_labels, NUM_CLASSES)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    eval_labels = keras.utils.to_categorical(eval_labels, NUM_CLASSES)

    # reshape images
    # To have input as an image, we reshape images beforehand.
    train_data = train_data.reshape(train_data.shape[0], INPUT_SHAPE)
    eval_data = eval_data.reshape(eval_data.shape[0], INPUT_SHAPE)

    # Create a keras model
    model = get_keras_model()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    # Convert the keras model to estimator
    classifier = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=FLAGS.model_dir)

    # Train Model
    input_name = model.input_names[0]
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={input_name: train_data},
        y=train_labels,
        batch_size=64,
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=FLAGS.steps)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={input_name: eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # Save the model
    classifier.export_savedmodel(FLAGS.saved_dir,
                                 serving_input_receiver_fn=serving_input_receiver_fn)


if __name__ == '__main__':
    tf.app.run()
