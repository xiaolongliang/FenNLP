#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
def loss(name,from_logits=False,label_smoothing=0,binary=True):
    if name=="CE":
        if binary:
            loss_op = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits,label_smoothing=label_smoothing)
        else:
            loss_op = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits,label_smoothing=label_smoothing)
        return loss_op

    if name=="MSE":
        loss_op = tf.keras.losses.MeanSquaredError()
        return loss_op

