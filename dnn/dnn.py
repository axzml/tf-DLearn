import warnings
warnings.filterwarnings('ignore')
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.python.ops import init_ops

import input_data

def generate_dataset(data_file, num_epochs, shuffle, batch_size):
    dataset = input_data.input_fn(data_file, num_epochs, shuffle, batch_size)
    dataset = dataset.make_one_shot_iterator().get_next()
    return dataset

class DeepModel(object):
    def __init__(self, units, hidden_units, activation_fn, drop_rate=None,):
        super(DeepModel, self).__init__()
        self.units = units
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn

        self.seq = []
        for num_hidden_unit in self.hidden_units:
            self.seq.append(layers.Dense(num_hidden_unit,
                                      kernel_initializer=init_ops.glorot_uniform_initializer(),
                                      activation=self.activation_fn))
            if drop_rate is not None:
                self.seq.append(layers.Dropout(drop_rate))
            self.seq.append(layers.BatchNormalization())
        self.seq.append(layers.Dense(units,
                                  kernel_initializer=init_ops.glorot_uniform_initializer(),
                                  activation=None))

    def __call__(self, x, training=True):
        for m in self.seq:
            if isinstance(m, layers.BatchNormalization) or \
                isinstance(m, layers.Dropout):
                x = m(x, training=training)
            else:
                x = m(x)
        return x

def accuracy(preds, labels):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc

train_file = '../Dataset/adult.data'
test_file = '../Dataset/adult.test'
num_epochs = 3
batch_size = 20
max_epochs = 2000
max_num_batches = 1000
learning_rate = 0.01
num_classes = 2

wide_columns, deep_columns = input_data.build_model_columns()
train_data = generate_dataset(train_file, num_epochs, True, batch_size)
test_data = generate_dataset(test_file, num_epochs, False, batch_size)

features, labels = train_data
labels = tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes)

x = tf.feature_column.input_layer(features, deep_columns)

units = 2
hidden_units = [10, 10]
activation_fn = tf.nn.relu
net = DeepModel(units, hidden_units, activation_fn, drop_rate=None,)
logits = net(x, training=True)
pred = tf.nn.softmax(logits, name='pred')

tf.set_random_seed(123)
global_step = tf.get_variable(
    'global_step',
    [],
    initializer=tf.constant_initializer(0),
    trainable=False)

with tf.variable_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

with tf.variable_scope('train'):
    grad_op = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = grad_op.minimize(cost, global_step=global_step)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(tf.tables_initializer())
#     print(sess.run(tf.shape(out)))
    try:
        for epoch in range(max_epochs):
            losses = []
            for _ in range(max_num_batches):
                p, l = tf.argmax(pred, 1), tf.argmax(labels, 1)
                acc = accuracy(pred, labels)
                loss, _, p1, l1, acc1, g = sess.run([cost, train_op, p, l, acc, global_step])
                print('Loss: {}, pred: {}, label: {}, acc: {:.4f}, global_step: {}'.format(loss, p1, l1, acc1, g))
            break
    except tf.errors.OutOfRangeError as e:
        print('Training Finished!')
