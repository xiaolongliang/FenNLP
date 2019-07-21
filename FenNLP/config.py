#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
import json
import copy
class Config(object):
    "Configuration for BertModel."
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size = 3072,
                 hidden_act = "gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02
                 ):
        self.vocab_size = vocab_size,
        self.hidden_size = hidden_size,
        self.num_hidden_layers=num_hidden_layers,
        self.num_attention_heads = num_attention_heads,
        self.intermediate_size=intermediate_size,
        self.hidden_act = hidden_act,
        self.hidden_dropout_prob = hidden_dropout_prob,
        self.attention_probs_dropout_prob = attention_probs_dropout_prob,
        self.max_position_embeddings = max_position_embeddings,
        self.type_vocab_size = type_vocab_size ,
        self.initializer_range = initializer_range
    @classmethod
    def from_dict(cls,json_object):
        config = Config(vocab_size=None)
        for (key,value) in json_object.items():
            config.__dict__[key] = value

    @classmethod
    def from_json_file(cls,json_file):
        with tf.gfile.GFile(json_file,"r") as reader:
            text = reader.read()
        return cls.from_dict(json.load(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(),indent=2,sort_keys=True)+'\n'




