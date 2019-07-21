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
                 is_training=True,shuffle=True,
                 mode="classfication",
                 data_path = "datasets",
                 vocab_file = "vocabs/vocab.txt"):
        self.data_name = data_name
        self.batch_size=batch_size
        self.epoch=epoch
        self.maxlen=maxlen
        self.seed=seed
        self.use_mask = use_mask
        self.use_segment = use_segment
        self.is_training=is_training
        self.shuffle=shuffle
        self.mode=mode
        self.data_path = data_path
        self.vocab_file = vocab_file
        self.do_lower_case =do_lower_case
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

    def dump_data(self,sentence,mode):
        path = os.path.join(self.data_path, self.data_name + '.npz')
        if mode=="classfication":
            self.tokenizer.convert_tokens_to_ids(tokens)# tokens is a list


            # data = {"x_train":[],"y_train":[],"x_test":[],"y_test":[],"x_mask_train":[],"x_mask_test":[],"x_seg_test":[]}
            data = {"x_train":[],"y_train":[],"x_test":[],"y_test":[],"x_mask_train":[],"x_mask_test":[],"x_seg_test":[]}
            #
            np.save(path,data)

    def padding(self,):
        pass
        #
    def __call__(self,is_training):
        path =os.path.join(self.data_path,self.data_name + '.npz')

        x_mask_train=None
        x_mask_test=None
        x_seg_train=None
        x_seg_test = None
        with np.load(path) as f:
            x_train, labels_train = f['x_train'], f['y_train']
            x_test, labels_test = f['x_test'], f['y_test']
            if self.use_mask:
                x_mask_train=f['x_mask_train']
                x_mask_test=f['x_mask_test']
            if self.use_segment:
                x_seg_train = f['x_seg_train']
                x_seg_test= f['x_seg_test']
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

loader = DataLoader("imdb", 10, 100, is_training=True,
                    use_mask=False,use_segment=False,
                    mode="classfication")

for input,target,mask,segment in loader(is_training=True):
    print(input)













