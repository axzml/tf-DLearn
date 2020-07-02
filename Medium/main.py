# _*_ cooding:utf-8 _*_
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import shutil
from sklearn.metrics import roc_auc_score

import tensorflow as tf

from model.LR import LR
from model.WideDeep import WideDeep

def train(model, train_iterator,
          max_epochs=1, max_num_batches=3,
          checkpoint='ckpt', save_suffix='model.ckpt',
          summary_save_path='summary', save_freq=10,
          log_freq=100):
    save_path = os.path.join(checkpoint, save_suffix)
    Xi, Xv, y = train_iterator
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(summary_save_path, graph=sess.graph)
        try:
            for epoch in range(max_epochs):
                for _ in range(max_num_batches):
                    loss, summary, global_step = model.fit_on_batch(sess, Xi, Xv, y)
                    writer.add_summary(summary, global_step)
                    if global_step % log_freq == 0:
                        print('global_step: {}, output: {}'.format(global_step, loss))
                    if global_step % save_freq == 0:
                        model.save_model(sess, save_path, global_step)
        except tf.errors.OutOfRangeError as e:
            print('Training Finished!')


def test(model, test_iterator, checkpoint='ckpt'):
    Xi, Xv, y = test_iterator
    predictions = []
    labels = []
    batch_num = 0
    with tf.Session() as sess:
        model.load_model(sess, checkpoint)
        while True:
            try:
                out, label = model.predict(sess, Xi, Xv, y)
                batch_num += 1
                predictions.append(out)
                labels.append(label)
                print('Processing Batch: {}, {}, {}'.format(batch_num, len(out), batch_num * len(out)))
            except tf.errors.OutOfRangeError as e:
                print('Testing Finished!')
                break

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(predictions.shape, labels.shape)
    score = roc_auc_score(labels, predictions)
    print('Auc Score: {}'.format(score))


def main():
    from data.dataset import build_adult_dataset
    train_data, test_data, params = build_adult_dataset()
    epoch = 1
    batch_size = 128
    max_epochs = 1000
    max_num_batches = 100000
    checkpoint = 'ckpt'
    save_suffix = 'model.ckpt'
    summary_save_path = 'summary'
    save_freq = 100
    log_freq = 100
    ### params for LR
    output_size = 1
    embedding_size = 128
    random_seed = 1234
    learning_rate = 0.01
    max_to_keep = 4
    model_name = 'WideDeep'
    ## params for WideDeep
    hidden_units = [128, 64]
    use_bn = True
    batch_norm_decay = 0.995
    use_linear = False
    use_deep = True
    activation = tf.nn.relu
    dropout_ratios = [.5, .5]
    rm_cache = True
    feature_size = params['feature_size']
    field_size = params['field_size']

    if rm_cache and os.path.exists(checkpoint):
        shutil.rmtree(checkpoint, ignore_errors=True)

    if rm_cache and os.path.exists(summary_save_path):
        shutil.rmtree(summary_save_path, ignore_errors=True)

    train_iterator = train_data.repeat(epoch).batch(batch_size).shuffle(1024) \
        .make_one_shot_iterator().get_next()
    test_iterator = test_data.batch(1024) \
        .make_one_shot_iterator().get_next()

    if model_name == 'LR':
        model = LR(feature_size, field_size,
                   output_size=output_size,
                   embedding_size=embedding_size,
                   random_seed=random_seed,
                   learning_rate=learning_rate,
                   max_to_keep=max_to_keep)
    elif model_name == 'WideDeep':
        model = WideDeep(feature_size, field_size,
                         output_size=output_size,
                         embedding_size=embedding_size,
                         random_seed=random_seed,
                         learning_rate=learning_rate,
                         max_to_keep=max_to_keep,
                         hidden_units=hidden_units,
                         use_bn=use_bn,
                         batch_norm_decay=batch_norm_decay,
                         use_linear=use_linear,
                         use_deep=use_deep,
                         activation=activation,
                         dropout_ratios=dropout_ratios,
                         )

    train(model, train_iterator, max_epochs=max_epochs,
          max_num_batches=max_num_batches,
          checkpoint=checkpoint,
          save_suffix=save_suffix,
          summary_save_path=summary_save_path,
          save_freq=save_freq,
          log_freq=log_freq)
    test(model, test_iterator, checkpoint=checkpoint)


if __name__ == '__main__':
    main()
