# _*_ cooding:utf-8 _*_
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from model.model import Model

class LR(Model):
    def __init__(self,
                 feature_size,
                 field_size,
                 output_size=1,
                 embedding_size=4,
                 random_seed=1234,
                 learning_rate=0.1,
                 max_to_keep=4,
                 ):
        super(LR, self).__init__()
        self.random_seed = random_seed
        self.feature_size = feature_size  # denote as M, size of the feature dictionary
        self.field_size = field_size  # denote as F, size of the feature fields
        self.output_size = output_size
        self.embedding_size = embedding_size  # denote as K, size of the feature embedding
        self.learning_rate = learning_rate
        self.max_to_keep = max_to_keep
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

        self._init_weights()

        self.embeddings = tf.nn.embedding_lookup(self.embeddings_dict,
                                                 self.feat_index)  # B * F * K
        feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
        self.embeddings = tf.multiply(self.embeddings, feat_value)  # B * F * K
        self.embeddings = tf.reduce_mean(self.embeddings, axis=1)  # B * K

        ## Model
        self.out = tf.matmul(self.embeddings, self.weight) + self.bias

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

    def _init_weights(self):
        with tf.variable_scope("embeddings"):
            self.embeddings_dict = tf.get_variable("feat_emb",
                                                   [self.feature_size, self.embedding_size],  # M * K
                                                   initializer=tf.random_normal_initializer(0.0, 0.1),
                                                   dtype=tf.float32)

            self.weight = tf.get_variable("weight",
                                          [self.embedding_size, self.output_size],
                                          initializer=tf.random_normal_initializer(0.0, 0.1),
                                          dtype=tf.float32)
            self.bias = tf.get_variable("bias",
                                        [self.output_size],
                                        initializer=tf.constant_initializer(0.0),
                                        dtype=tf.float32)

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
