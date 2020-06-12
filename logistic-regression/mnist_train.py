import warnings
warnings.filterwarnings('ignore')
import numpy as np
from os.path import join

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
save_model = False
save_root = 'mnist_model'
save_suffix = 'model.ckpt'
save_path = join(save_root, save_suffix)

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

def accuracy(preds, labels):
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc

def train():
    tf.set_random_seed(123)
    global_step = tf.get_variable(
        'global_step',
        [],
        initializer=tf.constant_initializer(0),
        trainable=False)

    features = tf.placeholder(tf.float32, [None, emb_dim], name='features')
    label = tf.placeholder(tf.float32, [None, num_classes], name='label')

    model = LR(emb_dim, num_classes, use_bias=True)

    logits = model(features)
    pred = tf.nn.softmax(logits, name='pred')

    with tf.variable_scope('loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    with tf.variable_scope('train'):
        grad_op = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = grad_op.minimize(cost, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=4)

    tf.summary.scalar('loss', cost)
    summary_op = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    count = 0
    with tf.Session() as sess:
        sess.run(init_op)
        writer = tf.summary.FileWriter(save_root, graph=sess.graph)
        try:
            for epoch in range(max_epochs):
                losses = []
                for _ in range(max_num_batches):
                    p, l = tf.argmax(pred, 1), tf.argmax(label, 1)
                    x, y = mnist.train.next_batch(batch_size)
                    acc = accuracy(pred, label)
                    loss, _, summary, p1, l1, acc1, g = sess.run([cost, train_op, summary_op, p, l, acc, global_step],
                                                feed_dict={features: x, label: y})
                    print('Loss: {}, pred: {}, label: {}, acc: {:.4f}, global_step: {}'.format(loss, p1, l1, acc1, g))
                    writer.add_summary(summary, g)
                    count += 1
                    if g % 10 == 0:
                        saver.save(sess, save_path, global_step=global_step, write_meta_graph=True)
                        break
                    # if count > 20:
                        # break
                break
        except tf.errors.OutOfRangeError as e:
            print('Training Finished!')
    print(count)


def test():
    with tf.Session() as sess:
        try:
            latest_checkpoint = tf.train.latest_checkpoint(save_root)
            meta_file = join(save_root, latest_checkpoint.split('/')[-1] + '.meta')
            new_saver = tf.train.import_meta_graph(meta_file)
            new_saver.restore(sess, latest_checkpoint)
            graph = tf.get_default_graph()
            features = graph.get_tensor_by_name('features:0')
            label = graph.get_tensor_by_name('label:0')
            pred = graph.get_tensor_by_name('pred:0')
            count, total = 0, 0
            for _ in range(10):
                x, y = mnist.train.next_batch(batch_size)
                feed_dict = {features: x, label: y}
                prediction, labels = sess.run([pred, label], feed_dict=feed_dict)
                correct_num = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
                count += tf.reduce_sum(tf.cast(correct_num, tf.float32)).eval()
                total += prediction.shape[0]
            print('Accuracy: {:.4f}%'.format(count / total * 100))
        except tf.errors.OutOfRangeError as e:
            print('Testing Finished!')

if __name__ == '__main__':
    train()
    test()
