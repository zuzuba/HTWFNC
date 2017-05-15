import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
weights = W.eval()
bias = np.atleast_2d(b.eval())

weights = np.vstack((weights, np.tile(bias, (2, 1))))


n_points = 1000
np.savetxt('data/weight.csv', weights, delimiter=', ', newline='\n')

x_test = np.hstack((mnist.test.images[:n_points, :], 0.5 *
                    np.ones((n_points, 2))))

np.savetxt('data/x_test.csv', x_test, delimiter=', ',
           newline='\n')
np.savetxt('data/y_test.csv', mnist.test.labels[:1000, :], delimiter=', ',
           newline='\n')


# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
