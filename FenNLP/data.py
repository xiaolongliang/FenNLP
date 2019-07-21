#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from FenNLP.tokenizer import FullTokenizer
import numpy as np
import os
tf.enable_eager_execution()

_datas = ["imdb", ]


class DataLoader(object):
    def __init__(self, data_name,
                 use_mask=False,
                 use_segment=False,
                 shuffle=True,
                 use_bert_tag=False,
                 token_pad_type="[PAD]",
                 data_path="datasets",
                 vocab_file="vocabs/vocab.txt",
                 seed=113,):
        self.data_name = data_name

        self.seed = seed
        self.token_pad_type = token_pad_type
        self.use_mask = use_mask
        self.use_segment = use_segment

        self.shuffle = shuffle

        self.data_path = data_path
        self.vocab_file = vocab_file

        self.use_bert_tag = use_bert_tag



    def rerange(self, x, labels=None, mask=None, segment=None):
        # np.random.seed(seed)
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x = x[indices]
        if self.use_mask:
            mask = mask[indices]
        if self.use_segment:
            segment = segment[indices]
        labels = labels[indices]
        return x, labels, mask, segment

    def map_label_to_id(self,num_class):
        label2id = {}
        for i in range(num_class):
            label2id["_label{}_".format(str(i))] = i
        return label2id

    def pares_data(self, path,maxlen,do_lower_case,num_class, mode="CLF"):
        tokenizer = FullTokenizer(vocab_file=self.vocab_file, do_lower_case=do_lower_case)
        x = []
        y = []
        mask = []
        segment = []
        label2id = self.map_label_to_id(num_class)
        if mode == "CLF":
            with open(path, 'r') as rf:
                for line in rf:
                    contents = line.strip().split('\t')
                    sentence = tokenizer.tokenize(contents[-1])
                    label = contents[0]
                    if self.use_bert_tag:
                        sentence = ["[CLS]"] + sentence
                        sentence = sentence + ["[SEP]"]
                    mask_ids = [1] * len(sentence)
                    sentence = self.padding(sentence, token_type=self.token_pad_type, max_length=maxlen)
                    token_ids = tokenizer.convert_tokens_to_ids(sentence)  # tokens is a list
                    label_ids = label2id[label]
                    segment_ids = [1]*len(sentence)
                    mask_ids.extend([0] * (len(sentence) - len(mask_ids)))

                    x.append(token_ids)
                    y.append(label_ids)
                    mask.append(mask_ids)
                    segment.append(segment_ids)
            return np.array(x), np.array(y), np.array(mask), np.array(segment)

        elif mode == "NER":
            pass

    def dump_data(self,maxlen,do_lower_case,num_class,mode):
        train_path = os.path.join(self.data_path, self.data_name + ".train")
        dev_path = os.path.join(self.data_path, self.data_name + ".dev")
        test_path = os.path.join(self.data_path, self.data_name + ".test")
        if os.path.exists(train_path) and os.path.exists(dev_path) and os.path.exists(test_path):
            x_train, y_train, x_mask_train, x_seg_train = self.pares_data(train_path, maxlen,do_lower_case,num_class,mode)
            x_dev, y_dev, x_mask_dev, x_seg_dev = self.pares_data(dev_path, maxlen,do_lower_case,num_class,mode)
            x_test, y_test, x_mask_test, x_seg_test = self.pares_data(test_path, maxlen,do_lower_case,num_class,mode)
            output_path = os.path.join(self.data_path, self.data_name)
            np.savez(output_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                     x_mask_train=x_mask_train, x_mask_test=x_mask_test,
                     x_seg_train=x_seg_train, x_seg_test=x_seg_test,
                     x_dev=x_dev, y_dev=y_dev,
                     x_mask_dev=x_mask_dev, x_seg_dev=x_seg_dev)
        else:
            raise FileNotFoundError("Must provide `name.train`,`name.dev`,`name.test`")

    def padding(self, sentence, token_type, max_length=128):
        length = len(sentence)
        if length < max_length:
            sentence.extend([token_type] * (max_length - length))
        else:
            sentence = sentence[:max_length]
        return sentence

    def from_data_sclice_pack(self, x, y, mask, segment,epoch,batch_size,use_mask=True,use_segment=True ):
        if not use_mask and use_segment:
            x, y, mask, segment = self.rerange(x, y, segment)
            loader = tf.data.Dataset.from_tensor_slices((x, y, segment)). \
                shuffle(buffer_size=len(x)).repeat(epoch).batch(batch_size)
            return loader
        elif not use_segment and use_mask :
            x, y, mask, segment = self.rerange(x, y, mask)
            loader = tf.data.Dataset.from_tensor_slices((x, y, mask)). \
                shuffle(buffer_size=len(x)).repeat(epoch).batch(batch_size)
            return loader
        elif not use_mask and not use_segment:
            x, y, mask, segment = self.rerange(x, y)
            loader = tf.data.Dataset.from_tensor_slices((x, y)). \
                shuffle(buffer_size=len(x)).repeat(epoch).batch(batch_size)
            return loader
        else:
            # print(x)
            # print(y)
            # print(mask)
            # print(segment)
            x, y, mask, segment = self.rerange(x, y, mask, segment)
            loader = tf.data.Dataset.from_tensor_slices((x, y, mask, segment)). \
                shuffle(buffer_size=len(x)).repeat(epoch).batch(batch_size)
            return loader

    def load(self,config, dev=False):
        self.dump_data(config.maxlen,config.do_lower_case,config.num_class,config.mode)
        path = os.path.join(self.data_path, self.data_name + '.npz')

        x_mask_train = None
        x_mask_test = None
        x_seg_train = None
        x_seg_test = None
        with np.load(path) as f:
            x_train, labels_train = f['x_train'], f['y_train']
            x_dev, labels_dev = f['x_dev'], f['y_dev']
            x_test, labels_test = f['x_test'], f['y_test']
            if self.use_mask:
                x_mask_train = f['x_mask_train']
                x_mask_dev = f['x_mask_dev']
                x_mask_test = f['x_mask_test']
            if self.use_segment:
                x_seg_train = f['x_seg_train']
                x_seg_dev = f['x_seg_dev']
                x_seg_test = f['x_seg_test']
        np.random.seed(self.seed)
        # xs = np.concatenate([x_train, x_test])
        # labels = np.concatenate([labels_train, labels_test])

        if config.is_training:
            if dev:
                loader = self.from_data_sclice_pack(x_dev, labels_dev, x_mask_dev, x_seg_dev,
                                                    config.epoch,config.batch_size,self.use_mask,self.use_segment)
                return loader
            else:
                loader = self.from_data_sclice_pack(x_train, labels_train, x_mask_train, x_seg_train,
                                                    config.epoch, config.batch_size,self.use_mask,self.use_segment)
                return loader

        else:
            loader = self.from_data_sclice_pack(x_test, labels_test, x_mask_test,x_seg_test,
                                                config.epoch, config.batch_size,self.use_mask,self.use_segment)
        return loader

def main():
    from FenNLP.config import ModelConfig
    config = ModelConfig(epoch=1, maxlen=128, lr=5e-5,
                         bert_config_file="cased_L-12_H-768_A-12/bert_config.json",
                         use_one_hot_embeddings=True, do_valid=True)

    loader = DataLoader("imdb")

    for input, target,_,_ in loader.load(config):
        print(input)

if __name__=="__main__":
    main()
