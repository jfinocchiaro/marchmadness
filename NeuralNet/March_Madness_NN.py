import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import argparse
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split


def shuffle(images, labels):
    all_data = list(zip(images, labels))
    random.shuffle(all_data)
    x, y = zip(*all_data)
    images = np.array(x)
    labels = np.array(y)
    return images, labels


display_step = 100
NUM_INPUT = 94
NUM_CLASSES = 1
baseline_accuracy = 0.5  # if you were to guess, this assumes an even distribution of the data
H1 = 2048
H2 = 2048
H3 = 2048

LEARNING_RATE = 0.01
train_step = 100000
BATCH_SIZE = 1200

# Store layers weight & bias
weights = {
    'wd1': tf.Variable(tf.random_normal([NUM_INPUT, H1], stddev=float(2) / NUM_INPUT ** 0.5)),
    'wd2': tf.Variable(tf.random_normal([H1, H2], stddev=float(2) / H1 ** 0.5)),
    'wd3': tf.Variable(tf.random_normal([H2, H3], stddev=float(2) / H2 ** 0.5)),
    'wd4': tf.Variable(tf.random_normal([H3, NUM_CLASSES], stddev=float(2) / H3 ** 0.5))
}

biases = {
    'bd1': tf.Variable(tf.constant(0.001, shape=[H1])),
    'bd2': tf.Variable(tf.constant(0.001, shape=[H2])),
    'bd3': tf.Variable(tf.constant(0.001, shape=[H3])),
    'bd4': tf.Variable(tf.constant(0.001, shape=[NUM_CLASSES]))
}

# tf Graph input
x = tf.placeholder(tf.float32, [None, NUM_INPUT])
y = tf.placeholder(tf.float32, [None, NUM_CLASSES])
keep_probability = tf.placeholder(tf.float32)

# Construct model
drop_0 = tf.nn.dropout(x, keep_probability)

fc1 = tf.add(tf.matmul(drop_0, weights['wd1']), biases['bd1'], name='fully_connected1')
fc1_nl = tf.nn.relu(fc1, name='fully_connected1_nl')
drop_1 = tf.nn.dropout(fc1_nl, keep_probability)

fc2 = tf.add(tf.matmul(drop_1, weights['wd2']), biases['bd2'], name='fully_connected2')
fc2_nl = tf.nn.relu(fc2, name='fully_connected2_nl')
drop_2 = tf.nn.dropout(fc2_nl, keep_probability)

fc3 = tf.add(tf.matmul(drop_2, weights['wd3']), biases['bd3'], name='fully_connected3')
fc3_nl = tf.nn.relu(fc3, name='fully_connected3_nl')
drop_3 = tf.nn.dropout(fc3_nl, keep_probability)

fc4 = tf.add(tf.matmul(drop_1, weights['wd4']), biases['bd4'], name='fully_connected4')

pred = fc4
guess = tf.round(tf.nn.sigmoid(pred))

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate model
correct_pred = tf.equal(guess, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

training_accuracies = []
validation_accuracies = []
dropout_rates = []
train_steps = []

all_input = pickle.load(open('../AdaBoost/pickled_files/all_train_x.p', 'rb'))
all_results = pickle.load(open('../AdaBoost/pickled_files/all_train_y.p', 'rb'))

training_data, rest_data, training_labels, rest_labels = train_test_split(all_input, all_results, test_size=0.5,
                                                                          random_state=None)
size_of_train = len(training_data)

training_data = np.array(training_data)
training_labels = np.split(np.array(training_labels, dtype='f'), len(training_labels))

validation_data, test_data, validation_labels, test_labels = train_test_split(rest_data, rest_labels, test_size=0.5,
                                                                              random_state=None)

size_of_val = len(validation_data)

validation_data = np.array(validation_data)
validation_labels = np.split(np.array(validation_labels, dtype='f'), len(validation_labels))

test_data = np.array(test_data)
test_labels = np.split(np.array(test_labels, dtype='f'), len(test_labels))


saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    dropout_rate = 1.0
    best_validation = 0

    for train_step in range(1, train_step + 1):
        end_batch_train_num = int(size_of_train / BATCH_SIZE)
        end_batch_val_num = int(size_of_val / BATCH_SIZE)
        batch_step_train = train_step % (end_batch_train_num)
        batch_step_val = train_step % (end_batch_val_num)

        if batch_step_train == end_batch_train_num - 1:
            training_data, training_labels = shuffle(training_data, training_labels)

        batch_x = training_data[BATCH_SIZE * batch_step_train:BATCH_SIZE * (batch_step_train + 1)]
        batch_y = training_labels[BATCH_SIZE * batch_step_train:BATCH_SIZE * (batch_step_train + 1)]

        batch_x_val = validation_data[BATCH_SIZE * batch_step_val:BATCH_SIZE * (batch_step_val + 1)]
        batch_y_val = validation_labels[BATCH_SIZE * batch_step_val:BATCH_SIZE * (batch_step_val + 1)]

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_probability: dropout_rate})

        loss, training_accuracy = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_probability: 1.0})
        validation_accuracy = sess.run(accuracy, feed_dict={x: batch_x_val, y: batch_y_val,
                                                            keep_probability: 1.0})  # set the keep_probability to keep all the nodes during vailidation
        if validation_accuracy >= best_validation:
            best_validation = validation_accuracy
            saver.save(sess, './my-model') #saves the best weights we have seen

        average_accuracy = (training_accuracy + validation_accuracy) / 2.0 #so the dropout formula is a bit easier to read

        dropout_rate = (dropout_rate + float(
            1 - ((max(0, (training_accuracy - baseline_accuracy) / float(1 - baseline_accuracy))) * min(1, abs(
                training_accuracy - validation_accuracy) / float(
                1 - average_accuracy))))) / 2.0  # average of old dropout rate and the proposed one


        dropout_rates.append(dropout_rate)
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)

        if train_step == 1 or train_step % display_step == 0:
            print(
                'Training Step= ' + str(train_step) + ', Minibatch Loss= ' + str(loss) + ', Training Accuracy= ' + str(
                    training_accuracy) + ', Validation Accuracy= ' + str(
                    validation_accuracy) + ', Keep Probability= ' + str(
                    dropout_rate))

        if train_step > 2000 or validation_accuracy > 0.95:
            break

    print('Optimization Finished')
    # saver.restore(sess, '/weights/best_weights.ckpt')  # restore the best session weights
    new_saver = tf.train.import_meta_graph('./my-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))

    test_accuracy = sess.run(accuracy, feed_dict={x: test_data, y: test_labels, keep_probability: 1.0})
    print('Testing accuracy= ' + str(test_accuracy))


