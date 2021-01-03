"""
train model
"""
import logging
import torch
import os
from os.path import join

logger = logging.getLogger("gpt2chat")
from LCCCDataGenerator import LCCCDataGenerator
from GPT2ChatbotConf import GPT2ChatbotConf
from transformers import BertTokenizer, AdamW, WarmupLinearSchedule, GPT2Config
from GPT2.GPT2Model import GPT2Model as GPT2LMHeadModel


def train(conf: GPT2ChatbotConf):
    logger.info("make dirs")
    conf.save_model_dir = join(conf.output_dir, conf.save_model_dir)
    if not os.path.exists(conf.save_model_dir):
        os.makedirs(conf.save_model_dir)
    ###
    logger.info("get train data")
    tokenizer = BertTokenizer.from_pretrained(join(conf.data_model_dir, conf.pretrained_model_dir))
    train_data = LCCCDataGenerator(conf, tokenizer)
    ###
    logger.info("get pretrained model")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(join(conf.data_model_dir, conf.pretrained_model_dir)).to(device)
    model.train()
    # 设置优化器，并且在初始训练时，使用warmup策略
    optimizer = AdamW(model.parameters(), lr=conf.lr, correct_bias=True)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=2000, t_total=train_data.steps * conf.epoch)

    ###
    logger.info("start train")
    for epoch in range(conf.epoch):
        for step, batch in enumerate(train_data):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["labels"] = batch["input_ids"]
            loss = model(**batch)[0]
            loss.backward()
            # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数
            optimizer.step()
            # 清空梯度信息
            optimizer.zero_grad()
            # 进行warm up
            scheduler.step()
            if step % conf.log_step == 0:
                logger.info("{}/{}:{}".format(step, train_data.steps, loss.data))

            if conf.save_step > 0 and step % conf.save_step == 0 and step > 0:
                save_dir = os.path.join(conf.save_model_dir, "{}-{}".format(epoch + 1, step))
                os.makedirs(save_dir)
                logger.info("save model to: {}".format(save_dir))
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
        if conf.save_step < 0:
            save_dir = os.path.join(conf.save_model_dir, "epoch-{}".format(epoch + 1))
            os.makedirs(save_dir)
            logger.info("save model to: {}".format(save_dir))
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
