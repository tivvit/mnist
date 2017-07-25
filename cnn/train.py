from mnist import MNIST
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import time


def cnn(x, use_dropout=True, dropout_prob=None):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_size=[5, 5],
        padding="same")

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same")

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024,
                            activation=tf.nn.relu)

    if use_dropout:
        dropout = tf.layers.dropout(inputs=dense, rate=dropout_prob)
        y_out = tf.layers.dense(inputs=dropout, units=10)
    else:
        y_out = tf.layers.dense(inputs=dense, units=10)

    return y_out


def dense(x):
    dropout = tf.layers.dropout(inputs=x, rate=0.5)
    l1 = tf.layers.dense(inputs=dropout, units=784)
    d2 = tf.layers.dropout(inputs=l1, rate=0.5)
    l2 = tf.layers.dense(inputs=d2, units=784)
    l3 = tf.layers.dense(inputs=l2, units=784)
    l4 = tf.layers.dense(inputs=l3, units=400)
    y_out = tf.layers.dense(inputs=l4, units=10)

    return y_out


def main(_):
    train, test = load()
    train_x, train_y = np.array(train[0]), np.array(train[1])
    test_x, test_y = np.array(test[0]), np.array(test[1])

    print("train len {}, test len {}".format(len(train_x), len(test_x)))
    print("Per digit stats {}".format(Counter(train_y)))

    train_y_oh = OneHotEncoder(n_values=10).fit_transform(
        train_y.reshape(-1, 1)).toarray()
    test_y_oh = OneHotEncoder(n_values=10).fit_transform(
        test_y.reshape(-1, 1)).toarray()

    print("Training cnn")
    test_acc = train_model(test_x, test_y_oh, train_x, train_y_oh, "cnn")
    print('Test accuracy for cnn {}'.format(test_acc))
    print("Training dense")
    test_acc = train_model(test_x, test_y_oh, train_x, train_y_oh, "dense",
                           iterations=20000)
    print('Test accuracy for dense {}'.format(test_acc))


def train_model(test_x, test_y_oh, train_x, train_y_oh, model_name,
                iterations=10000):
    BATCH_SIZE = 100
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        dropout_ratio = tf.placeholder(tf.float32)

        if model_name == "cnn":
            model_out = cnn(x, dropout_prob=dropout_ratio)
        if model_name == "dense":
            model_out = dense(x)

        with tf.name_scope('loss'):
            cross_entropy = \
                tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                        logits=model_out)
        cross_entropy = tf.reduce_mean(cross_entropy)
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(model_out, 1),
                                          tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        with tf.name_scope('summaries'):
            tf.summary.scalar('acc', accuracy)
            tf.summary.scalar('loss', cross_entropy)

        # store the graph and the metrics
        graph_location = "/app/log/{}-{}".format(model_name, time.time())
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location,
                                             graph=tf.get_default_graph())
        summaries = tf.summary.merge_all()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            epoch = 0
            steps = 0
            train_x, train_y_oh = shuffle_data(train_x, train_y_oh)
            for i in range(iterations):
                batch = get_batch(BATCH_SIZE, steps, train_x, train_y_oh)
                steps += 1
                if len(batch[0]) < BATCH_SIZE:
                    # todo wrap-up should be implemented here
                    # it is not necessary because we are shuffling the data
                    epoch += 1
                    steps = 0
                    train_x, train_y_oh = shuffle_data(train_x, train_y_oh)
                    batch = get_batch(BATCH_SIZE, steps, train_x, train_y_oh)

                if i % 200 == 0:
                    train_accuracy, summary = sess.run(
                        [accuracy, summaries],
                        feed_dict={x: batch[0], y_: batch[1],
                                   dropout_ratio: 1.0})
                    train_writer.add_summary(summary, i)
                    print('Epoch {}: step {} training accuracy {}'.format(
                        epoch, i, train_accuracy))

                train_step.run(
                    feed_dict={x: batch[0], y_: batch[1], dropout_ratio: 0.5})

            test_acc = accuracy.eval(
                feed_dict={x: test_x, y_: test_y_oh, dropout_ratio: 1.0})

            return test_acc


def get_batch(step, steps, train_x, train_y_oh):
    start = steps * step
    end = (steps + 1) * step
    batch = [train_x[start:end], train_y_oh[start:end]]
    return batch


def shuffle_data(train_x, train_y_oh):
    rnd = np.arange(len(train_x))
    np.random.shuffle(rnd)
    train_x = train_x[rnd]
    train_y_oh = train_y_oh[rnd]
    return train_x, train_y_oh


def load():
    # this is may be done with
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets("/data", one_hot=True)
    print("LOADING DATA")
    mndata = MNIST('/data')
    train = mndata.load_training()
    test = mndata.load_testing()
    print("DATA LOADED")

    return train, test


if __name__ == "__main__":
    tf.app.run(main)
