import numpy as np
import os
import tensorflow as tf

output_file = './data/mnist.tfrecord'

# with tf.python_io.TFRecordWriter(output_file) as writer:
    # for _ in range(10):
        # label = np.random.randint(0, 10)
        # img = np.array([np.random.randint(0, 255) for _ in range(16)], dtype=np.uint8)

        # example = tf.train.Example(features=tf.train.Features(feature={
            # "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
            # "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
        # }))

        # writer.write(example.SerializeToString())


def read_and_decode(filenames, batch_size=2, num_epochs=None, perform_shuffle=False):
    def _parse_fn(record):
        features = {
            "label": tf.FixedLenFeature([], tf.int64),
            "image": tf.FixedLenFeature([], tf.string),
        }
        parsed = tf.parse_single_example(record, features)
        # image
        image = tf.decode_raw(parsed["image"], tf.uint8)
        image = tf.reshape(image, [16,])
        # label
        label = tf.cast(parsed["label"], tf.int64)
        return {"image": image}, label

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    # multi-thread pre-process then prefetch
    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=10).prefetch(500000)

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def input_fn(libsvm_file, batch_size=2, num_epochs=1, perform_shuffle=False):
    def _parse_fn(line):
        # line = tf.Print(line, [line], message='PRINT: ')
        columns = tf.string_split([line], ' ')
        # rows = tf.sparse_tensor_to_dense(columns, default_value='')
        # rows = tf.Print(rows, [rows], message='Rows: ', summarize=100)
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')  # filed_size=280 feature_size=6500000
        # cols = tf.sparse_tensor_to_dense(splits, default_value='')
        # cols = tf.Print(cols, [cols, tf.shape(cols)], message='Cols: ', summarize=100)
        # splits = tf.Print(splits, [splits], message='PRINT: ')
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        # id_vals = tf.Print(id_vals, [id_vals], message='PRINT: ', summarize=100)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        # feat_ids = tf.Print(feat_ids, [tf.shape(feat_ids)])
        # feat_vals = tf.sign(feat_vals) * tf.math.log(tf.abs(feat_vals) + 1)  # do log manual
        # return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels
        return {"feat_ids": feat_ids}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(libsvm_file).map(_parse_fn, num_parallel_calls=10).prefetch(500000)

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)  # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


# batch_features, batch_labels = read_and_decode("./data/mnist.tfrecord")
# with tf.Session() as sess:
    # print(sess.run(batch_features["image"][0]))
    # print(sess.run(batch_labels[0]))

batch_features, batch_labels = input_fn("./data/libsvm.file.txt")
with tf.Session() as sess:
    print(sess.run(batch_features))
    print(sess.run(batch_labels))
