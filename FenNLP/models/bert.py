#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import copy
from FenNLP.embedding import *
from FenNLP.transformer import *
class BERTModel(tf.keras.Model):
    def __init__(self,
                 config=None,
                 is_training=None,
                 use_one_hot_embeddings=False,
                 scope=None):

        self.config = config
        self.is_training = is_training
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.scope=scope
        super(BERTModel,self).__init__()
    def call(self,input_ids,input_mask=None,token_type_ids=None):

        config = copy.deepcopy(self.config)
        if not self.is_training:
            config.hidden_dropout_prob=0.0
            config.attention_probs_dropout_prob = 0.0
        input_shape = input_ids.shape.to_list()
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size,seq_length],dtype=tf.int32)
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size,seq_length],dtype=tf.int32)
        with tf.variable_scope(self.scope,default_name='bert'):
            with tf.variable_scope("embeddings"):
                # token embedding
                token_embedding = WDEmbedding(vocab_size=config.vocab_size,
                                              embedding_size=config.hidden_size,
                                              initializer_range=config.initializer_range,
                                              word_embedding_name="word_embeddings",
                                              use_one_hot_embedding=self.use_one_hot_embeddings)

                (self.embedding_output,self.embedding_table) = token_embedding(input_ids)

                #segment and position embedding
                segposembedding = SegPosEmbedding(use_token_type=True,
                                                  token_type_ids=token_type_ids,
                                                  token_type_vocab_size=config.type_vocab_size,
                                                  token_type_embedding_name="token_type_embeddings",
                                                  use_position_embeddings=True,
                                                  position_embedding_name="position_embeddings",
                                                  initializer_range=config.initializer_range,
                                                  max_position_embeddings=config.max_position_embeddings,
                                                  drop_prob=config.hidden_dropout_prob)

                self.embedding_output = segposembedding(self.embedding_output)

                with tf.variable_scope("encoder"):
                    attention_mask = create_attention_mask_from_input_mask(input_ids,input_mask)

                    encoder_layers = Transformer(hidden_size=config.hidden_size,
                                                 num_hidden_layers=config.num_hidden_layers,
                                                 num_attention_heads=config.num_attention_heads,
                                                 intermediate_size=config.intermediate_size,
                                                 intermediate_act_fn=get_activation(config.hidden_act),
                                                 hidden_dropout_prob=config.attention_probs_dropout_prob,
                                                 do_return_all_layers=True,
                                                 initializer_range=config.initializer_range
                                                 )
                    self.all_encoder_layers = encoder_layers(self.embedding_output,attention_mask)

                self.sequence_output = self.all_encoder_layers[-1]
                with tf.variable_scope("pooler"):
                    first_token_tensor = tf.squeeze(self.sequence_output[:,0:1,:],axis=1)
                    self.pooled_output = tf.keras.layers.Dense(
                        config.hidden_size,
                        activation=tf.tanh,
                        dtype=float,
                        kernel_constraint=create_initializer(config.initializer_range)
                    )(first_token_tensor)
    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table
