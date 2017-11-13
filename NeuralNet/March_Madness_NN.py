import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pickle

# Parameters
learning_rate = 0.005
epochs = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 512
n_hidden_2 = 256
teamfeaturevectors = 50
num_classes = 1

# tf Graph input
X = tf.placeholder("float", [None, teamfeaturevectors])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'X': tf.Variable(tf.random_normal([teamfeaturevectors, n_hidden_1])),

    'h1': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),

    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'X': tf.Variable(tf.random_normal([n_hidden_1])),
    'h1': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def prepare(raw, queries):
    data = []
    labels = []
    count = 0
    for tup in queries:
        count += 1
        if count == 3000:
            break
        season = tup[0]
        teamANum = tup[1]
        teamBNum = tup[2]
        teamAResult= tup[3]

        teamAFeatures = raw[season][teamANum]
        teamBFeatures = raw[season][teamBNum]

        X = np.concatenate((teamAFeatures,teamBFeatures))

        data.append(X)
        labels.append([teamAResult])
    return data, labels

def neural_net():
    # Create model
    # Hidden fully connected layer with some number of neurons
    layer_X = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['X']), biases['X']))

    layer_hidden1 = tf.nn.sigmoid(tf.add(tf.matmul(layer_X, weights['h1']), biases['h1']))

    # Output fully connected layer with a neuron for each class
    # out_layer = tf.nn.sigmoid(tf.matmul(layer_hidden, weights['out']) + biases['out'])
    return tf.matmul(layer_hidden1, weights['out']) + biases['out']

# Define loss and optimizer
logits = neural_net()
prediction =  tf.round(tf.nn.sigmoid(logits))

loss_op = tf.losses.mean_squared_error(logits, Y)
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
# correct_pred = tf.reduce_all(tf.equal(tf.round(out_layer), out_person))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
correct_pred = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    raw = pickle.load(open('decay_True_normalized_feature_vec.p','rb'))
    queries = pickle.load(open('season_tuples.p','rb'))
    data,labels = prepare(raw, queries)

    eighty_percent_split = int(len(data)*0.8)

    train_x = data[:eighty_percent_split]
    train_y = labels[:eighty_percent_split]
    test_x = data[eighty_percent_split:]
    test_y = labels[eighty_percent_split:]

    for step in range(1, epochs + 1):
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: train_x, Y: train_y})
        # print(sess.run(forus, feed_dict={X: train_x, Y: train_y}))
        # print(sess.run(prediction, feed_dict={X: train_x, Y: train_y}))
        # quit()
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: train_x,
                                                                 Y: train_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))