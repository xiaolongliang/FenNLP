#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""
import tensorflow as tf
import collections
import re
version = tf.__version__
Version_float = float('.'.join(version.split('.')[:2]))

def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.truncated_normal(stddev=initializer_range)

def dropout(tensor, drop_out_rate):
    if drop_out_rate is None or drop_out_rate == 0.0:
        return tensor
    output = tf.nn.dropout(tensor, 1.0 - drop_out_rate)
    return output

def reshape_to_matrix(tensor):
    if len(tensor.shape) == 0:
        return tensor
    dim = tensor.shape[-1]
    tensor_2d = tf.reshape(tensor, [-1, dim])
    return tensor_2d

def reshape_from_matrix(output_tensor,orig_shape_list):
    if len(orig_shape_list)==2:
        return output_tensor
    output_shape = output_tensor.shape.to_list()
    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]
    return tf.reshape(output_tensor,orig_dims+[width])

def layer_norm(input_tensor,name=None):
    if Version_float>1.13:
        return tf.keras.layers.LayerNormalization(axis=-1,name=name)(input_tensor)
    else:
        return tf.layers.BatchNormalization(axis=-1,name=name)(input_tensor)

def layer_norm_and_dropout(input_tensor,dropout_prob,name=None):
    output_tensor = layer_norm(input_tensor,name)
    output_tensor = dropout(output_tensor,dropout_prob)
    return output_tensor

def create_attention_mask_from_input_mask(from_tensor,to_mask):
    """
    Create 3D attention mask from a 2D tensor mask.
    from_tensor [B,F,D]
    to_mask [B,T]
    """
    assert_rank(from_tensor,[2,3])
    from_shape = from_tensor.shape.to_list()
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = to_mask.shape.to_list()
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask,[batch_size,1,to_seq_length]),tf.float32
    )

    broadcast_ones = tf.ones(
        shape=[batch_size,from_seq_length,1],dtype=tf.float32
    )

    mask = broadcast_ones*to_mask
    return mask

def assert_rank(tensor,expected_rank,name=None):
    if name is None:
        name = tensor.name
    excepted_rank_dict = {}
    if isinstance(expected_rank,int):
        excepted_rank_dict[expected_rank]=True
    else:
        for x in expected_rank:
            excepted_rank_dict[x]=True
    actual_rank = tensor.shape.ndims
    if actual_rank not in excepted_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise(
            "For tensor {} in scope {}, the actual rank {} is not equal"
            "to expected rank {}".format(name,scope_name,actual_rank,str(expected_rank))
        )

def gelu(x):
    import numpy as np
    cdf = 0.5*(1.0+tf.tanh((np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))
    return x*cdf

def get_activation(activation_string):
    "map string to activation function"
    if not isinstance(activation_string,str):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()

    if act=="linear":
        return None
    elif act=="gelu":
        return gelu
    elif act=="relu":
        return tf.nn.relu
    elif act=="tanh":
        return tf.nn.tanh
    else:
        raise ValueError("Unsupport activation:%s" % act)

def get_assignment_map_from_checkpoint(tvars,init_checkpoint):
    assignment_map = {}
    initialized_variable_names = {}
    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$",name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()

    for x in init_vars:
        (name,var) = (x[0],x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] =name
        initialized_variable_names[name]=1
        initialized_variable_names[name+":0"]=1
    return (assignment_map, initialized_variable_names)











