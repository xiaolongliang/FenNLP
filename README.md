# FenNLP 

An out-of-the-box NLP toolkit can easily help you solve tasks such as entity recognition, relationship extraction, text categorization and so on.

# This is the early version of FenNLP. Update 2019/7/21

# Status

* Still on the development （not complete）

# Usage (example for bert model)
```python

import tensorflow as tf
from FenNLP.models.bert import BERTModel
from FenNLP.trainers.trainer import Trainer
from FenNLP.optim import Optimizer
from FenNLP.losses import loss
from FenNLP.data import DataLoader
from FenNLP.config import BertConfig,ModelConfig
tf.enable_eager_execution()

# model config
config = ModelConfig(epoch=1,maxlen=128,lr=5e-5,
                     bert_config_file="cased_L-12_H-768_A-12/bert_config.json",
                     use_one_hot_embeddings=True,do_valid=True)

# Data
loader = DataLoader("imdb",use_mask=True, use_segment=True, shuffle=True, use_bert_tag=True)

# Model
class Model(tf.keras.Model):
    def __init__(self, config, use_one_hot_embeddings, scope=None):
        self.bert_config_file = config
        self.use_one_hot_embedding = use_one_hot_embeddings
        self.scope = scope
        self.bert_config = BertConfig.from_json_file(json_file=self.bert_config_file)
        super(Model, self).__init__()

    def call(self, inputs, masks=None,
             segments=None, is_training=True):
        model = BERTModel(config=self.bert_config,
                          is_training=is_training,
                          use_one_hot_embeddings=self.use_one_hot_embeddings,
                          scope=None)
        m = model(inputs, masks, segments)
        # bert_output = m.get_sequence_output()#返回序列
        bert_output = m.get_pooled_output()  # 返回分类向量
        return bert_output


model = Model(config.bert_config_file, use_one_hot_embeddings=config.use_one_hot_embeddings)

# Optimizer
optimizer = Optimizer('adam', config.lr)

# losser
loss = loss('CE', from_logits=True, label_smoothing=True)

# Train and valid
Trainer.train(
    loader=loader,
    model=model,  # 模型
    optimizer=optimizer,  # 优化器
    loss=loss,  # 损失函数
    config =config)  # 是否使用预训练权重

```