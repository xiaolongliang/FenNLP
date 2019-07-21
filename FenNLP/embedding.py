#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
@Tensorflow 2.0
All of the following Code was follow Google BERT!
"""
import tensorflow as tf
from FenNLP.tools import create_initializer, layer_norm_and_dropout

class WDEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size,
                 embedding_size=128,
                 initializer_range=0.02,
                 word_embedding_name="word_embeddings",
                 use_one_hot_embedding=False):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.initializer_range = initializer_range
        self.word_embedding_name = word_embedding_name
        self.use_one_hot_embedding = use_one_hot_embedding

        super(WDEmbedding, self).__init__()

    def build(self, input_shape):
        self.embedding_table = self.add_variable(
            name=self.word_embedding_name,
            dtype=float,
            shape=[self.vocab_size, self.embedding_size],
            initializer=create_initializer(self.initializer_range)
        )

    def call(self, input_ids):
        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])
        flat_input_ids = tf.reshape(input_ids, [-1])
        if self.use_one_hot_embedding:
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=self.vocab_size)
            output = tf.linalg.matmul(one_hot_input_ids, self.embedding_table)
        else:
            output = tf.gather(self.embedding_table, flat_input_ids)
        input_shape = list(input_ids.shape)
        output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * self.embedding_size])
        return output, self.embedding_table


class SegPosEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 use_token_type=False,
                 token_type_ids=None,
                 token_type_vocab_size=16,
                 token_type_embedding_name="token_type_embeddings",
                 use_position_embeddings=True,
                 position_embedding_name="position_embeddings",
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 drop_prob=0.1):

        super(SegPosEmbedding, self).__init__()
        self.use_token_type = use_token_type
        self.token_type_ids = token_type_ids
        self.token_type_vocab_size = token_type_vocab_size
        self.token_type_embedding_name = token_type_embedding_name
        self.use_position_embeddings = use_position_embeddings
        self.position_embedding_name = position_embedding_name
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.drop_prob = drop_prob

    def build(self, input_shape):
        self.token_type_table = self.add_variable(
            name=self.token_type_embedding_name,
            shape=[self.token_type_vocab_size, input_shape[2]],
            dtype=float,
            initializer=create_initializer(self.initializer_range)
        )
        self.full_position_embeddings = self.add_variable(
            name=self.position_embedding_name,
            shape=[self.max_position_embeddings, input_shape[2]],
            dtype=float,
            initializer=create_initializer(self.initializer_range)
        )

    def call(self, input_tensor):
        inputshape = input_tensor.shape.as_list()
        batch_size = inputshape[0]
        seq_length = inputshape[1]
        width = inputshape[2]
        output = input_tensor
        # segment features
        if self.use_token_type:
            if self.token_type_ids is None:
                raise ValueError("token_type_ids must be specified if use_token_type is True")
            flat_token_type_ids = tf.reshape(self.token_type_ids, [-1])
            one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
            token_type_embeddings = tf.linalg.matmul(one_hot_ids, self.token_type_table)
            token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])
            output += token_type_embeddings
        # position features
        if self.use_position_embeddings:
            assert_op = tf.assert_less_equal(seq_length, self.max_position_embeddings)
            with tf.control_dependencies([assert_op]):
                # actual sequence length migth be shorter than this,so we use slice ti help.
                position_embeddings = tf.slice(self.full_position_embeddings, [0, 0], [seq_length, -1])
                num_dims = len(output.shape.as_list())
                position_broadcast_shape = []
                for _ in range(num_dims - 2):
                    position_broadcast_shape.append(1)
                position_broadcast_shape.extend([seq_length, width])
                position_embeddings = tf.reshape(position_embeddings,
                                                 position_broadcast_shape)
                output += position_embeddings

        output = layer_norm_and_dropout(output, self.drop_prob)
        return output



# TODO
class ELMOEmbedding(tf.keras.Model):
    def __init__(self,model:str="original",options_file:str=None,weight_file:str=None):

        # import allennlp.commands.elmo
        super(ELMOEmbedding,self).__init__()
        pass




