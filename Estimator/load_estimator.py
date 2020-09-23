# 需要 tensorflow 2.0
import tensorflow as tf


export_dir = './pb_model/1600870178'

estimator = tf.saved_model.load(
    export_dir=export_dir
)

print(type(estimator))
print(dir(estimator))

f = estimator.signatures['prediction']
print(f(tf.constant([[0, 0, 0], [2, 2, 2], [3, 1, 6]])))
