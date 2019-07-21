#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
import json
import copy
class BertConfig(object):
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
        config = BertConfig(vocab_size=None)
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

class ModelConfig(object):
    def __init__(self,epoch = 1,
                lr = 5e-5,
                maxlen = 128,
                batch_size = 200,
                num_class = 2,
                use_one_hot_embeddings = True,
                init_checkpoint = "cased_L-12_H-768_A-12/bert_model.ckpt",
                bert_config_file = "cased_L-12_H-768_A-12/bert_config.json",
                do_lower_case = False,
                is_training = True,
                do_valid = True,
                data_name = "imdb",
                mode = "CLF"):
        self.epoch = epoch
        self.lr = lr
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.num_class = num_class
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.init_checkpoint = init_checkpoint
        self.bert_config_file = bert_config_file
        self.do_lower_case =do_lower_case
        self.is_training =is_training
        self.do_valid = do_valid
        self.data_name = data_name
        self.mode = mode

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


def main():
    config = ModelConfig()
    x = config.to_dict()
    print(x)

if __name__=="__main__":
    main()



