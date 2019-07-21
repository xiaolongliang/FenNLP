#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
from FenNLP.tokenizer import FullTokenizer
import numpy as np
import os
_datas=["imdb",]
class DataLoader(object):
    def __init__(self,data_name,batch_size,
                 epoch, do_lower_case=False,maxlen=None,
                 seed=113, use_mask=True,
                 use_segment = True,
                 is_training=True,
                 shuffle=True,
                 use_bert_tag = True,
                 num_class=2,
                 token_pad_type = "[PAD]",
                 mode="classfication",
                 data_path = "datasets",
                 vocab_file = "vocabs/vocab.txt"):
        self.data_name = data_name
        self.batch_size=batch_size
        self.epoch=epoch
        self.maxlen=maxlen
        self.seed=seed
        self.token_pad_type = token_pad_type
        self.use_mask = use_mask
        self.use_segment = use_segment
        self.is_training=is_training
        self.shuffle=shuffle
        self.mode=mode
        self.data_path = data_path
        self.vocab_file = vocab_file
        self.do_lower_case =do_lower_case
        self.use_bert_tag = use_bert_tag
        self.num_class=num_class
        self.tokenizer = FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

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
    def map_label_to_id(self):
        label2id = {}
        for i in range(self.num_class):
            label2id["_lable{}_".format(str(i))]=i
        return label2id
    def pares_data(self,path,mode="classfication"):
        x=[]
        y=[]
        mask=[]
        segment=[]
        if mode == "classfication":
            with open(path, 'r') as rf:
                for line in rf:
                    contents = line.strip().split('\t')
                    sentence = self.tokenizer.tokenize(contents[-1])
                    label = contents[0]
                    if self.use_bert_tag:
                        sentence = ["[CLS]"] + sentence
                        sentence = sentence+["[SEP]"]
                    mask = [1]*len(sentence)
                    sentence = self.padding(sentence, token_type=self.token_pad_type, max_length=self.maxlen)
                    token_ids = self.tokenizer.convert_tokens_to_ids(sentence)  # tokens is a list
                    label_ids = self.map_label_to_id()[label]
                    segment_ids=1
                    mask_ids = mask.extend([0]*(len(sentence)-len(mask)))
                    x.append(token_ids)
                    y.append(label_ids)
                    mask.append(mask_ids)
                    segment.append(segment_ids)
            return np.array(x),np.array(y),np.array(mask),np.array(segment)

        elif mode=="NER":
            pass


    def dump_data(self):
        train_path = os.path.join(self.data_path, self.data_name+".train")
        dev_path = os.path.join(self.data_path, self.data_name+".dev")
        test_path = os.path.join(self.data_path, self.data_name+".test")
        if os.path.exists(train_path) and os.path.exists(dev_path) and os.path.exists(test_path):
            x_train,y_train,x_mask_train,x_seg_train = self.pares_data(train_path, mode=self.mode)
            x_dev,y_dev,x_mask_dev,x_seg_dev = self.pares_data(dev_path, mode=self.mode)
            x_test,y_test,x_mask_test,x_seg_test = self.pares_data(test_path, mode=self.mode)
            data = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test, "x_mask_train": x_mask_train,
                    "x_mask_test": x_mask_test,"x_seg_train": x_seg_train, "x_seg_test": x_seg_test,"x_dev":x_dev,"y_dev":y_dev,
                    "x_mask_dev":x_mask_dev,"x_seg_dev":x_seg_dev}
            output_path = os.path.join(self.data_path, self.data_name, ".npz")
            np.save(output_path, data)
            del data
        else:
            raise FileNotFoundError("Must provide `name.train`,`name.dev`,`name.test`")

    def padding(self,sentence,token_type,max_length=128):
        length = len(sentence)
        if length< max_length:
            sentence.extend([token_type]*(max_length-length))
        else:
            sentence = sentence[:max_length]
        return sentence

    def __call__(self,is_training):
        path =os.path.join(self.data_path,self.data_name + '.npz')

        x_mask_train=None
        x_mask_test=None
        x_seg_train=None
        x_seg_test = None
        with np.load(path) as f:
            x_train, labels_train = f['x_train'], f['y_train']
            x_test, labels_test = f['x_dev'], f['y_dev']
            if self.use_mask:
                x_mask_train=f['x_mask_train']
                x_mask_test=f['x_mask_dev']
            if self.use_segment:
                x_seg_train = f['x_seg_train']
                x_seg_test= f['x_seg_dev']
        np.random.seed(self.seed)
        # xs = np.concatenate([x_train, x_test])
        # labels = np.concatenate([labels_train, labels_test])
        print(x_train)
        print(labels_train)
        print(x_mask_train)
        print(x_seg_train)
        if is_training:
            x_train, labels_train, x_mask_train, x_seg_train = self.rerange(x_train, labels_train,
                                                                            x_mask_train, x_seg_train)
            loader = tf.data.Dataset.from_tensor_slices((x_train,labels_train,x_mask_train,x_seg_train)
                                                        ).shuffle(buffer_size=len(x_train)).repeat(self.epoch).batch(self.batch_size)
        else:
            x_test, labels_test, x_seg_train, x_seg_test = self.rerange(x_test, labels_test,
                                                                        x_mask_test, x_seg_test)
            loader = tf.data.Dataset.from_tensor_slices((x_test,labels_test,x_mask_test,x_seg_test)
                                                        ).batch(self.batch_size)

        return loader

loader = DataLoader("imdb", batch_size=10,epoch=1, maxlen=128, is_training=True,
                    use_mask=False,use_segment=False, mode="classfication")

for input,target,mask,segment in loader(is_training=True):
    print(input)













