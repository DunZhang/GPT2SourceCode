# GPT2SourceCode

## 简介
为了加深对GPT2理解，决定手动实现GPT2并在LCCC数据集上训练一个闲聊机器人

## 运行方法
修改`GPT2ChatbotConf.py`中的参数，然后运行`run_train.py`即可训练模型

## 相关说明
- 本库主要是为了帮助广大网友加深对GPT2的理解而开源的，工程化的东西比较少，偏实验性质一些
- 麻雀虽小，五脏俱全，支持加载基于`transformers`的预训练模型(我只测试了LCCC的NovelGPT,如果不兼容就自己改动下预训练权重名字)
- 实际测试和`transformers`中的`GPT2Model`输出一致，可以说是完全复现GPT2模型
- 代码简洁易于修改，基于本代码可以快速实现BERT和UniLM，以UniLM为例，只要改成下图的mask形式即可，ABCD可以理解为第一个句子，EFGH可以理解为待生成的句子，
这非常像把seq2seq的编码器和解码器写到了一起。

![UniLM.png](https://i.loli.net/2021/01/03/tpP5iyacR4hreq2.png)

## 重要代码文件说明
- `GPT2ChatbotConf.py` 训练闲聊机器人的配置文件
- `interact.py` 对话测试
- `LCCCDataGenerator.py` LCCC数据生成器，可开启多进程
- `Train.py` 训练代码
- `GPT2Layer.py` GPT2源码
- `MultiHeadSelfAttention.py` 多头自注意力机制



