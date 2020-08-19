import numpy as np
import os
import tensorflow as tf

def input_fn(libsvm_file, batch_size=2, num_epochs=1, perform_shuffle=False):
    def _parse_fn(line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')  # filed_size=280 feature_size=6500000
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids}, labels

    dataset = tf.data.TextLineDataset(libsvm_file).map(_parse_fn, num_parallel_calls=10).prefetch(500000)

    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


if __name__ == '__main__':
    batch_features, batch_labels = input_fn("./data/libsvm.file.txt")
    with tf.Session() as sess:
        print(sess.run(batch_features))
        print(sess.run(batch_labels))
