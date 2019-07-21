#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
def Optimizer(optim,lr=0.001):
    if optim=="adam":
        op = tf.keras.optimizers.Adam(
            lr=lr,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=None,
            decay=0.,
            amsgrad=False
        )
        return op
    elif optim == "rmsp":
        op = tf.keras.optimizers.RMSprop(
            lr=lr,
            rho=0.9,
            epsilon=None,
            decay=0.
        )
        return op




