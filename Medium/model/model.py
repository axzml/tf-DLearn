#_*_ cooding:utf-8 _*_
import warnings
warnings.filterwarnings('ignore')
import os

import tensorflow as tf


class Model(object):
    def __init__(self):
        super(Model, self).__init__()

    def _init_saver(self, *args, **kwargs):
        ## saver
        self.saver = tf.train.Saver(*args, **kwargs)

    def save_model(self, sess, save_path, global_step):
        self.saver.save(sess, save_path, global_step=int(global_step), write_meta_graph=True)
        print('Saving Model Successfully!')

    def load_model(self, sess, save_root):
        latest_checkpoint = tf.train.latest_checkpoint(save_root)
        meta_file = os.path.join(save_root, latest_checkpoint.split('/')[-1] + '.meta')
        print('Loading: {}'.format(meta_file))
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, latest_checkpoint)
        return sess