from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer('max_steps', 10000, 'The number of steps to train a model')
tf.app.flags.DEFINE_string('model_dir', './models/ckpt/', 'Dir to save a model and checkpoints')
FLAGS = tf.app.flags.FLAGS

INPUT_FEATURE = 'x'
NUM_CLASSES = 3


def serving_input_receiver_fn():
    """
    This is used to define inputs to serve the model.

    :return: ServingInputReciever
    """
    receiver_tensors = {
        'sepal_length': tf.placeholder(tf.float32, [None, 1]),
        'sepal_width': tf.placeholder(tf.float32, [None, 1]),
        'petal_length': tf.placeholder(tf.float32, [None, 1]),
        'petal_width': tf.placeholder(tf.float32, [None, 1]),
    }

    # Convert give inputs to adjust to the model.
    features = {
        INPUT_FEATURE: tf.concat([
            receiver_tensors['sepal_length'],
            receiver_tensors['sepal_width'],
            receiver_tensors['petal_length'],
            receiver_tensors['petal_width']
        ], axis=1)
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors,
                                                    features=features)


def main(_):
    # Load training and eval data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split data into train and eval.
    index_list = range(len(y))
    index_train, index_eval = train_test_split(index_list, train_size=0.8)
    X_train, X_eval = X[index_train], X[index_eval]
    y_train, y_eval = y[index_train], y[index_eval]

    # Define the feature columns for inputs.
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
        n_classes=NUM_CLASSES,
        model_dir=FLAGS.model_dir,
    )

    # Define training spec.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: X_train},
        y=y_train,
        # batch_size=5,
        num_epochs=1000,
        shuffle=True)
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=FLAGS.max_steps)

    # Define evaluating spec.
    latest_exporter = tf.estimator.LatestExporter(
        name="models",
        serving_input_receiver_fn=serving_input_receiver_fn,
        exports_to_keep=10)
    best_exporter = tf.estimator.BestExporter(
        serving_input_receiver_fn=serving_input_receiver_fn,
        exports_to_keep=1)
    exporters = [latest_exporter, best_exporter]
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE: X_eval},
        y=y_eval,
        num_epochs=1,
        shuffle=False)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        throttle_secs=180,
        steps=10,
        exporters=exporters)

    # Train and evaluate the model.
    tf.estimator.train_and_evaluate(classifier, train_spec=train_spec, eval_spec=eval_spec)


if __name__ == "__main__":
    tf.app.run()
