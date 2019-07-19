import tensorflow as tf


def masked_reduce_max(t: tf.Tensor, mask: tf.Tensor, axis=None,
                      keepdims=None, name=None) -> tf.Tensor:
    with tf.name_scope(name=name or 'masked_reduce_max'):
        min_mask = tf.where(mask, t, tf.ones_like(t) * MIN_POSSIBLE_VALUE)
        return tf.math.reduce_max(min_mask, axis=axis, keepdims=keepdims)


def masked_softmax(logits: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    logits_max = masked_reduce_max(logits, mask, axis=-1, keepdims=1)
    logtis_max = tf.stop_gradient(logits_max)
    normalized_logits = logits - logits_max
    zeros = tf.zeros_like(logits)
    normalized_exp = tf.where(mask, tf.math.exp(normalized_logits), zeros)
    normalized_exp_sum = tf.reduce_sum(normalized_exp, axis=-1, keepdims=True)
    normalized_log_sum = tf.math.log(normalized_exp_sum)
    return tf.where(mask, normalized_exp / normalized_exp_sum, zeros)
