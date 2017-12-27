import argparse
import os
import sys
import time
import json
import math

import tensorflow as tf
from six.moves import xrange
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.python.keras._impl.keras.utils.generic_utils import Progbar


FLAGS = None
CONFIG = None
CONFIG_PATH = './config.json'

class Config:

    def __init__(self):
        with open(CONFIG_PATH if CONFIG_PATH is not None else './config.json') as config_file:
            config_json = json.load(config_file)
            self.learning_rate = config_json["training_settings"]["learning_rate"]
            self.size_hidden_1 = config_json["training_settings"]["size_hidden_1"]
            self.size_hidden_2 = config_json["training_settings"]["size_hidden_2"]
            self.batch_size = config_json["training_settings"]["batch_size"]
            self.epoch = config_json["training_settings"]["epoch"]
            self.eval_every_n_steps = config_json["training_settings"]["eval_every_n_steps"]
        if not all(self.__dict__.values()):
            raise ValueError("Config initialization failed")

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(CONFIG.batch_size)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    true_count = 0
    steps_per_epoch = int(data_set.num_examples / CONFIG.batch_size)
    num_examples = steps_per_epoch * CONFIG.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                    images_placeholder,
                                    labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir)
    max_steps = math.ceil(CONFIG.epoch * data_sets.train.num_examples / CONFIG.batch_size)

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(
            CONFIG.batch_size)

        logits = mnist.inference(images_placeholder,
                                 CONFIG.size_hidden_1,
                                 CONFIG.size_hidden_2)

        # Add to the Graph the Ops for loss calculation.
        loss = mnist.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = mnist.training(loss, CONFIG.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        if FLAGS.c:
            saver.restore(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'))

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)
        progbar = Progbar(target=CONFIG.eval_every_n_steps)
        for step in xrange(max_steps):

            start_time = time.time()

            feed_dict = fill_feed_dict(data_sets.train,
                                        images_placeholder,
                                        labels_placeholder)

            _, loss_value = sess.run([train_op, loss],
                                    feed_dict=feed_dict)
            
            progbar.update((step % CONFIG.eval_every_n_steps) + 1, [("Loss", loss_value)], force=True)

            duration = time.time() - start_time

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % CONFIG.eval_every_n_steps == 0 or (step + 1) == max_steps:

                print("Total : ", int((step + 1) / CONFIG.eval_every_n_steps), "/", int(math.ceil(max_steps/CONFIG.eval_every_n_steps)))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                data_sets.test)

                progbar = Progbar(target=CONFIG.eval_every_n_steps)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir) and not FLAGS.c:
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        action='store_true',
        help='Wether or not to continue training on saved model.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join('.', 'tensorflow/mnist/input_data'),
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join('.', 'log'),
        help='Directory to put the log data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    CONFIG = Config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)