import warnings
warnings.filterwarnings('ignore')
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./Dataset/", one_hot=True)

num_classes = 10
batch_size = 20
num_epochs = 100
# num_parallel_calls = tf.data.experimental.AUTOTUNE
num_parallel_calls = 1
max_epochs = 2000
max_num_batches = 1000
learning_rate = 0.1
emb_dim = 784

class LR(object):
    def __init__(self, emb_dim, num_classes=1, use_bias=True):
        super(LR, self).__init__()
        self.emb_dim = emb_dim
        self.use_bias = use_bias
        with tf.variable_scope('weight'):
            W_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            self.W = tf.get_variable('W',
                                    shape=[emb_dim, num_classes],
                                    initializer=W_initer,
                                    dtype=tf.float32)

        with tf.variable_scope("bias"):
            b_initer = tf.constant(0, shape=[num_classes], dtype=tf.float32)
            self.b = tf.get_variable('b',
                                     dtype=tf.float32,
                                     initializer=b_initer)

    def __call__(self, x):
        x = tf.add(tf.matmul(x, self.W), self.b)
        return x

tf.set_random_seed(123)
global_step = tf.get_variable(
    'global_step',
    [],
    initializer=tf.constant_initializer(0),
    trainable=False)


features = tf.placeholder(tf.float32, [None, emb_dim])
label = tf.placeholder(tf.float32, [None, num_classes])

model = LR(emb_dim, num_classes, use_bias=True)

logits = model(features)
pred = tf.nn.softmax(logits)

with tf.variable_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

with tf.variable_scope('train'):
    grad_op = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = grad_op.minimize(cost, global_step=global_step)

init_op = tf.global_variables_initializer()
count = 0
with tf.Session() as sess:
    sess.run(init_op)
    try:
        for epoch in range(max_epochs):
            losses = []
            for _ in range(max_num_batches):
                p, l = tf.argmax(pred, 1), tf.argmax(label, 1)
                x, y = mnist.train.next_batch(batch_size)
                loss, _, p, l, g = sess.run([cost, train_op, p, l, global_step],
                                            feed_dict={features: x, label: y})
                print('Loss: {}, pred: {}, label: {}, global_step: {}'.format(loss, p, l, g))
                count += 1
                # if count > 20:
                    # break
            # break
    except tf.errors.OutOfRangeError as e:
        print('Training Finished!')
print(count)
