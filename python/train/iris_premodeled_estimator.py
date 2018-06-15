from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer('steps', 10000, 'The number of steps to train a model')
tf.app.flags.DEFINE_string('model_dir', './models/ckpt/', 'Dir to save a model and checkpoints')
tf.app.flags.DEFINE_string('saved_dir', './models/pb/', 'Dir to save a model for TF serving')
FLAGS = tf.app.flags.FLAGS

INPUT_FEATURE = 'x'
NUM_CLASSES = 3


def serving_input_receiver_fn():
    """
    This is used to define inputs to serve the model.

    :return: ServingInputReciever
    """
    reciever_tensors = {
        'sepal_length': tf.placeholder(tf.float32, [None, 1]),
        'sepal_width': tf.placeholder(tf.float32, [None, 1]),
        'petal_length': tf.placeholder(tf.float32, [None, 1]),
        'petal_width': tf.placeholder(tf.float32, [None, 1]),
    }

    # Convert give inputs to adjust to the model.
    features = {
        INPUT_FEATURE: tf.concat([
            reciever_tensors['sepal_length'],
            reciever_tensors['sepal_width'],
            reciever_tensors['petal_length'],
            reciever_tensors['petal_width']
        ], axis=1)
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                                                    features=features)


def main(_):
    # Load training and eval data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    index_list = range(len(y))
    index_train, index_test = train_test_split(index_list, train_size=0.8)
    X_train, X_test = X[index_train], X[index_test]
    y_train, y_test = y[index_train], y[index_test]

    # feature columns
    feature_columns = [
        tf.feature_column.numeric_column(INPUT_FEATURE, shape=[4])
    ]

    # Create the Estimator
    training_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps=100,
        save_checkpoints_steps=100)
    classifier = tf.estimator.DNNClassifier(
        config=training_config,
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        # optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=NUM_CLASSES,
        # dropout=0.8,
        model_dir=FLAGS.model_dir
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: X_train},
        y=y_train,
        # batch_size=5,
        num_epochs=1,
        shuffle=True)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=FLAGS.steps)

    # Evaluate the model and print results
    latest_exporter = tf.estimator.LatestExporter(
        name="models",
        serving_input_receiver_fn=serving_input_receiver_fn,
        exports_to_keep=10,
    )
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: X_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        throttle_secs=180,
        steps=10,
        exporters=latest_exporter)
    tf.estimator.train_and_evaluate(classifier, train_spec=train_spec, eval_spec=eval_spec)


if __name__ == "__main__":
    tf.app.run()
