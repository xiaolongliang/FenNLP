#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf

class Trainer(object):
    @staticmethod
    def train(loader,model, loss,optimizer,config):
        if config.is_training:
            for input,target,mask,segment in loader.load(is_training=True):
                if config.init_checkpoint!=None:
                    tvars = model.variables
                    (assignment_map, initialized_variable_names) = model.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                    tf.train.init_from_checkpoint(config.init_checkpoint, assignment_map)
                with tf.GradientTape() as tape:
                    logits = model(input, mask, segment,config.is_training)
                    loss, predict = loss(logits, target)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                if config.do_valid:
                    Trainer.dev(loader,model,loss,config)
        else:
            if config.do_valid:
                Trainer.dev(loader,model,loss,config)
            else:
                raise ValueError("`is_training` or `do_valid` one of them must be true!")
    @staticmethod
    def dev(loader,model,loss,config):
        for tinput, tmask, ttarget,tsegment in loader.load(config =config,dev =True):
            tlogits = model(tinput, tmask, tsegment, False)
            loss, _ = loss(tlogits, ttarget)
            _epoch = loader.epoch
            _batch = loader.batch
            print("EPOCH:{} LOSS:{}".format(_epoch, loss))

    def predict(self,):
        pass
