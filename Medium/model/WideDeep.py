# _*_ cooding:utf-8 _*_
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

from model.model import Model

class WideDeep(Model):
    def __init__(self,
                 feature_size,
                 field_size,
                 hidden_units=[128, 64],
                 use_bn=True,
                 batch_norm_decay=0.995,
                 output_size=1,
                 embedding_size=4,
                 random_seed=1234,
                 learning_rate=0.1,
                 max_to_keep=4,
                 use_linear=True,
                 use_deep=True,
                 activation=tf.nn.relu,
                 dropout_ratios=[.5, .5],
                 ):
        super(WideDeep, self).__init__()
        self.random_seed = random_seed
        self.feature_size = feature_size  # denote as M, size of the feature dictionary
        self.field_size = field_size  # denote as F, size of the feature fields
        self.hidden_units = hidden_units
        self.use_bn = use_bn
        self.batch_norm_decay = batch_norm_decay
        self.output_size = output_size
        self.embedding_size = embedding_size  # denote as K, size of the feature embedding
        self.learning_rate = learning_rate
        self.max_to_keep = max_to_keep
        self.use_linear = use_linear
        self.use_deep = use_deep
        self.activation = activation
        self.dropout_ratios = dropout_ratios

        self._init_graph()

    def _init_graph(self):
        # self.graph = tf.Graph()
        # with self.graph.as_default():
        tf.set_random_seed(self.random_seed)
        self.feat_index = tf.placeholder(tf.int32,
                                         shape=[None, self.field_size],
                                         name='feat_index')  # B * F
        self.feat_value = tf.placeholder(tf.float32,
                                         shape=[None, self.field_size],
                                         name='feat_value')  # B * F
        self.label = tf.placeholder(tf.int32,
                                    shape=[None, 1],
                                    name='label')  # B * 1
        self.is_training = tf.placeholder(tf.bool,
                                          name='is_training')

        weights = self._init_weights()

        self.embeddings = tf.nn.embedding_lookup(weights['embeddings_dict'],
                                                 self.feat_index)  # B * F * K
        feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
        self.embeddings = tf.multiply(self.embeddings, feat_value)  # B * F * K
        self.embeddings = tf.reduce_mean(self.embeddings, axis=1)  # B * K

        ## Model
        self.linear_out = tf.matmul(self.embeddings, weights['linear_weight']) + weights['linear_bias']
        for i in range(len(self.hidden_units)):
            self.deep_out = tf.matmul(self.embeddings, weights['dnn_weight_{}'.format(i)]) + \
                            weights['dnn_bias_{}'.format(i)]
            if self.use_bn:
                self.deep_out = self.batch_norm_layer(self.deep_out,
                                                      is_training=self.is_training,
                                                      scope_bn="bn_{}".format(i))
            self.deep_out = self.activation(self.deep_out)
            self.deep_out = tf.nn.dropout(self.deep_out, self.dropout_ratios[i])

        if self.use_linear and self.use_deep:
            final_input = tf.concat([self.linear_out, self.deep_out], axis=1)
        elif self.use_linear:
            final_input = self.linear_out
        elif self.use_deep:
            final_input = self.deep_out

        self.out = tf.matmul(final_input, weights['final_weight']) + weights['final_bias']

        ## Loss
        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.losses.log_loss(labels=self.label, predictions=self.out)

        ## Optimizer
        self.global_step = tf.get_variable('global_step',
                                           [],
                                           initializer=tf.constant_initializer(0),
                                           trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=0.9, beta2=0.999, epsilon=1e-8
                                                ).minimize(self.loss, global_step=self.global_step)
        ## Saver
        self._init_saver(max_to_keep=self.max_to_keep)

        ## summary
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

    def batch_norm_layer(self, x, is_training, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True,
                              updates_collections=None, is_training=True, reuse=None,
                              trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True,
                                  updates_collections=None, is_training=False, reuse=True,
                                  trainable=True, scope=scope_bn)
        z = tf.cond(is_training, lambda: bn_train, lambda: bn_inference)
        return z

    def _init_weights(self):
        weights = {}
        with tf.variable_scope("embeddings"):
            weights['embeddings_dict'] = tf.get_variable("embeddings_dict",
                                                       [self.feature_size, self.embedding_size],  # M * K
                                                       initializer=tf.random_normal_initializer(0.0, 0.1),
                                                       dtype=tf.float32)

        with tf.variable_scope("linear_weights"):
            weights['linear_weight'] = tf.get_variable("linear_weight",
                                                      [self.embedding_size, self.output_size],
                                                      initializer=tf.random_normal_initializer(0.0, 0.1),
                                                      dtype=tf.float32)
            weights['linear_bias'] = tf.get_variable("linear_bias",
                                                    [self.output_size],
                                                    initializer=tf.constant_initializer(0.0),
                                                    dtype=tf.float32)

        with tf.variable_scope("dnn_weights"):
            num_layer = len(self.hidden_units)
            input_size = self.embedding_size
            glorot = np.sqrt(2.0 / (input_size + self.hidden_units[0]))
            weights['dnn_weight_0'] = tf.get_variable("dnn_weight_0",
                                                      [self.embedding_size, self.hidden_units[0]],
                                                      initializer=tf.random_normal_initializer(0.0, glorot),
                                                      dtype=tf.float32)
            weights['dnn_bias_0'] = tf.get_variable("dnn_bias_0",
                                                    [1, self.hidden_units[0]],
                                                    initializer=tf.random_normal_initializer(0.0, glorot),
                                                    dtype=tf.float32)
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.hidden_units[i-1] + self.hidden_units[i]))
                weights['dnn_weight_{}'.format(i)] = tf.get_variable("dnn_weight_{}".format(i),
                                                                  [self.embedding_size, self.hidden_units[i]],
                                                                  initializer=tf.random_normal_initializer(0.0, glorot),
                                                                  dtype=tf.float32)
                weights['dnn_bias_{}'.format(i)] = tf.get_variable("dnn_bias_{}".format(i),
                                                                    [1, self.hidden_units[i]],
                                                                    initializer=tf.random_normal_initializer(0.0, glorot),
                                                                    dtype=tf.float32)

        ## final layer
        with tf.variable_scope('final_layer'):
            if self.use_linear and self.use_deep:
                input_size = self.output_size + self.hidden_units[-1]
            elif self.use_linear:
                input_size = self.output_size
            elif self.use_deep:
                input_size = self.hidden_units[-1]
            glorot = np.sqrt(2.0 / (input_size + 1))
            weights['final_weight'] = tf.get_variable('final_weight',
                                                     [input_size, 1],
                                                     initializer=tf.random_normal_initializer(0.0, glorot),
                                                     dtype=tf.float32)
            weights['final_bias'] = tf.get_variable('final_bias',
                                                    [1, 1],
                                                    initializer=tf.random_normal_initializer(0.0, glorot),
                                                    dtype=tf.float32)
        return weights

    def fit_on_batch(self, sess, Xi, Xv, y):
        if isinstance(Xi, tf.Tensor):
            Xi, Xv, y = sess.run([Xi, Xv, y])
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.is_training: True}
        loss, opt, summary, global_step = sess.run((self.loss,
                                                    self.optimizer,
                                                    self.summary_op,
                                                    self.global_step),
                                                   feed_dict=feed_dict)
        return loss, summary, int(global_step)

    def predict(self, sess, Xi, Xv, y=None):
        if isinstance(Xi, tf.Tensor):
            if y is not None:
                Xi, Xv, label = sess.run([Xi, Xv, y])
            else:
                Xi, Xv = sess.run([Xi, Xv])
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.is_training: False}
        out = sess.run(self.out, feed_dict=feed_dict)
        res = (out, label) if y is not None else out
        return res
