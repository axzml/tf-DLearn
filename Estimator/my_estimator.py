import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

def build_dnn_model(features, mode, params):
    feature_size = params['feature_size']
    embedding_size = params['embedding_size']
    weights = {key: tf.get_variable(key, [feature_size, embedding_size],
                                   initializer=tf.random_normal_initializer(0.0, 0.1),
                                   dtype=tf.float32) for key in features.keys()}
    ## embeddings 中每个元素的大小为 [B, field_size, 1, embedding_dim]
    embeddings = [
        tf.nn.embedding_lookup(weights[key], feature_index)
        for key, feature_index in features.items()
    ]

    embeddings = tf.concat(embeddings, axis=-1)
    batch_size  = tf.shape(embeddings)[0]
    embeddings = tf.reshape(embeddings, (batch_size, params['field_size'] * params['embedding_size']))

    x = embeddings
    for units in params['hidden_units']:
        x = tf.layers.dense(x, units, activation=tf.nn.relu)
        x = tf.layers.dropout(x, params['dropout_rate'], training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(x, 1, activation=None)
    return logits

class MyEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir='model',
                 params=None,
                 config=None,
                 warm_start_from=None,
                ):

        def model_fn(features, labels, mode, params):
            with tf.variable_scope('ctr_model'):
                logits = build_dnn_model(features, mode, params)
            pred= tf.sigmoid(logits, name="CTR")
            labels = tf.reshape(labels, (-1, 1))
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                          logits=logits),
                                                                          name="loss")
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'],
                                            beta1=0.9, beta2=0.999, epsilon=1e-8)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            auc = tf.metrics.auc(labels=labels, predictions=pred)
            metrics = {'auc': auc}
            tf.summary.scalar('auc', auc[1]) ## tf.metrics.auc 返回 (auc, update_op), 后者常用于测试集
            tf.summary.scalar('loss', loss)

            summary_hook = tf.train.SummarySaverHook(
                save_steps=1,
                output_dir=params['model_dir'],
                summary_op=tf.summary.merge_all(),
            )

            predictions = {
              'ctr_probabilities': pred,
            }
            export_outputs = {
              'prediction': tf.estimator.export.PredictOutput(predictions)
            }

            ## Predict, 最简单的情况
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                predictions=predictions,
                                                export_outputs=export_outputs)


            ## Evaluation
            if mode == tf.estimator.ModeKeys.EVAL:
                    return tf.estimator.EstimatorSpec(
                            mode=mode,
                            predictions=predictions,
                            loss=loss,
                            )

            ## 最后是 Train
            return tf.estimator.EstimatorSpec(
                            mode=mode,
                            predictions=predictions,
                            loss=loss,
                            train_op=train_op,
                            training_hooks=[summary_hook])


        super(MyEstimator, self).__init__(model_fn=model_fn,
                                          model_dir=model_dir,
                                          params=params,
                                          config=config,
                                          warm_start_from=warm_start_from)


from make_tfrecord import input_fn

params = {
    'hidden_units': [10, 5],
    'dropout_rate': 0.5,
    'feature_size': 10,
    'embedding_size': 4,
    'field_size': 3, 
    'learning_rate': 0.5,
    'model_dir': 'model',
}

train_files = "./data/libsvm.file.txt"
batch_size = 1
mode = tf.estimator.ModeKeys.TRAIN
model_dir = params['model_dir']

config = tf.estimator.RunConfig().replace(
        log_step_count_steps=1,
        save_summary_steps=1,
        save_checkpoints_secs=None,
        save_checkpoints_steps=1,
        keep_checkpoint_max=1
)
estimator = MyEstimator(model_dir=model_dir,
                       params=params,
                       config=config)
train_spec = tf.estimator.TrainSpec(
    input_fn=lambda: input_fn(train_files, batch_size),
    max_steps=1000,
)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(train_files, batch_size), throttle_secs=10, steps=None)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
