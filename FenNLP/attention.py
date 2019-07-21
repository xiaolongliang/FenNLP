#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:kaiyinzhou
@Tensorflow 2.0
All of the following Code was follow Google BERT!
"""

from __future__ import absolute_import, division, print_function
import math
import tensorflow as tf
from FenNLP.tools import create_initializer,dropout,reshape_to_matrix



def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
    output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

class AttentionLayer(tf.keras.Model):
    """
    Performs multi-headed attention from `from_tensor` to `to_tensor`.
    """

    def __init__(self,
                 attention_mask=None,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 num_attention_heads=1,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 batch_size=None,
                 from_seq_length=None,
                 to_seq_length=None):
        super(AttentionLayer, self).__init__()

        self.attention_mask = attention_mask
        self.size_per_head = size_per_head
        self.query_act = query_act
        self.key_act = key_act
        self.num_attention_heads = num_attention_heads
        self.value_act = value_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.do_return_2d_tensor = do_return_2d_tensor
        self.batch_size = batch_size
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length

    def init_component(self):
        # `query_layer` =[B*F, N*H]
        _query_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=self.query_act,
            name="query",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        # `value_layer` = [B*T, N*H]
        _key_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=self.key_act,
            name="key",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        # `query_layer` =[B*T, N*H]
        _value_layer = tf.keras.layers.Dense(
            self.num_attention_heads * self.size_per_head,
            activation=self.value_act,
            name="value",
            kernel_initializer=create_initializer(self.initializer_range)
        )
        return _query_layer, _key_layer, _value_layer

    def call(self, from_tensor, to_tensor):
        from_shape = from_tensor.shape.as_list()
        to_shape = to_tensor.shape.as_list()
        if len(from_shape) != len(to_shape):
            raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")
        if len(from_shape) == 3:
            self.batch_size = from_shape[0]
            self.from_seq_length = from_shape[1]
            self.to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (self.batch_size is None or self.from_seq_length is None or self.to_seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified."
                )
        _query_layer, _key_layer, _value_layer = self.init_component()

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)

        query_layer = _query_layer(from_tensor_2d)
        key_layer = _key_layer(to_tensor_2d)
        value_layer = _value_layer(to_tensor_2d)
        # [B,N,F,H]
        query_layer = transpose_for_scores(query_layer, self.batch_size, self.num_attention_heads,
                                           self.from_seq_length, self.size_per_head)
        # [B,N,T,H]
        key_layer = transpose_for_scores(key_layer, self.batch_size, self.num_attention_heads,
                                         self.to_seq_length, self.size_per_head)

        # attention_score = [B,N,F,T]
        attention_score = tf.linalg.matmul(query_layer, key_layer, transpose_b=True)
        attention_score = tf.math.multiply(attention_score, 1.0 / math.sqrt(float(self.size_per_head)))

        if self.attention_mask is not None:
            # [batch_size, from_seq_length, to_seq_length]
            attention_mask = tf.expand_dims(self.attention_mask, axis=[1])
            # Here we convert aim position to zero, masked position to -10000
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            # This could let score for masked very low
            attention_score += adder

        # sofmax
        attention_probs = tf.math.softmax(attention_score)
        attention_probs = dropout(attention_probs, self.attention_probs_dropout_prob)
        # value [B,N,T,H]
        value_layer = transpose_for_scores(value_layer, self.batch_size, self.num_attention_heads,
                                           self.to_seq_length, self.size_per_head)
        # context_layer [B,N,F,H]
        context_layer = tf.linalg.matmul(attention_probs, value_layer)
        # [B,F,N,H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if self.do_return_2d_tensor:
            context_layer = tf.reshape(context_layer, [self.batch_size * self.from_seq_length,
                                                       self.num_attention_heads * self.size_per_head])
        else:
            context_layer = tf.reshape(context_layer, [self.batch_size, self.from_seq_length,
                                                       self.num_attention_heads * self.size_per_head])
        return context_layer
