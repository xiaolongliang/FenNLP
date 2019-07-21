#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from FenNLP.tools import *
from FenNLP.attention import AttentionLayer

class Transformer(tf.keras.Model):
    def __init__(self,
                 hidden_size = 768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size = 3072,
                 intermediate_act_fn=gelu,
                 hidden_dropout_prob = 0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range =0.02,
                 do_return_all_layers=False
                 ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.do_return_all_layers = do_return_all_layers

        super(Transformer,self).__init__()

    def call(self,input_tensor,attention_mask):
        if self.hidden_size%self.num_attention_heads!=0:
            raise ValueError(
                "The hidden size must be the integer multiple of num attention heads"
            )
        attention_head_size = int(self.hidden_size/self.num_attention_heads)
        assert_rank(input_tensor,[3])
        input_shape = input_tensor.shape.to_list()
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]
        # due to the use of residuals here we should keep the dim same between input and hidden layer
        if input_width != self.hidden_size:
            raise ValueError(
                "The width of the input tensor {} != hidden size {}".format(input_width,self.hidden_size)
            )
        # keep representation as a 2Dtensor to avoid re-shaping it back and forth from a 3D tensor
        prev_output = reshape_to_matrix(input_tensor)

        all_layer_outputs = []
        for layer_idx in range(self.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output
                with tf.variable_scope("attention"):
                    attention_heads = []
                    with tf.variable_scope("self"):
                        attention_layer = AttentionLayer(
                            attention_mask=attention_mask,
                            num_attention_heads=self.num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                            initializer_range=self.initializer_range,
                            do_return_2d_tensor=True,
                            batch_size = batch_size,
                            from_seq_length=seq_length,
                            to_seq_length=seq_length
                        )
                        attention_head = attention_layer(layer_input,layer_input)
                        attention_heads.append(attention_head)
                    attention_output = None
                    if len(attention_heads)==1:
                        attention_output = attention_heads[0]
                    else:
                        attention_output = tf.concat(attention_heads,axis=-1)

                    # linear add residual add Batchnormalize
                    with tf.variable_scope("output"):
                        attention_output_layer = tf.keras.layers.Dense(
                            self.hidden_size,
                            dtype=float,
                            kernel_initializer=create_initializer(self.initializer_range)
                        )
                        attention_output = attention_output_layer(attention_output)
                        attention_output = dropout(attention_output,self.hidden_dropout_prob)
                        attention_output = layer_norm(attention_output+layer_input)
                    with tf.variable_scope("intermediate"):
                        intermediate_output = tf.keras.layers.Dense(
                            self.intermediate_size,
                            activation=self.intermediate_act_fn,
                            dtype=float,
                            kernel_initializer=create_initializer(self.initializer_range)
                        )(attention_output)

                    with tf.variable_scope("output"):
                        layer_output = tf.keras.layers.Dense(
                            self.hidden_size,
                            dtype=float,
                            kernel_initializer=create_initializer(self.initializer_range)
                        )(intermediate_output)
                        layer_output = dropout(layer_output,self.hidden_dropout_prob)
                        layer_output = layer_norm(layer_output+attention_output)
                        prev_output = layer_output
                        all_layer_outputs.append(layer_output)
            if self.do_return_all_layers:
                final_outputs=[]
                for layer_output in all_layer_outputs:
                    final_output = reshape_from_matrix(layer_output,input_shape)
                    final_outputs.append(final_output)
                return final_outputs
            else:
                final_output = reshape_from_matrix(prev_output,input_shape)
                return final_output












