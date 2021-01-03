"""
data genertator
"""
import re
import json
from os.path import join
import random
from GPT2ChatbotConf import GPT2ChatbotConf
from multiprocessing import Pool
import torch
from typing import List
from transformers import BertTokenizer, GPT2LMHeadModel
import time
import logging
from tqdm import tqdm

logger = logging.getLogger("gpt2chat")
SPEAKERS = ["[speaker1]", "[speaker2]"]


class LCCCDataGenerator():
    def __init__(self, conf: GPT2ChatbotConf, tokenizer: BertTokenizer):
        self.conf = conf
        self.tokenizer = tokenizer
        self.speaker_ids = tokenizer.convert_tokens_to_ids(["[speaker1]", "[speaker2]"])
        self.pool = Pool(1)
        # get all chatlog
        logger.info("read raw data...")
        self.chat_log = self._get_chatlog()
        logger.info("num data:{}".format(len(self.chat_log)))
        self.data_iter = iter(self.chat_log)
        self.steps = len(self.chat_log) // self.conf.batch_size
        # 创建一个数据进程
        if self.conf.use_multi_proc:
            batch_examples = self._get_batch_examples()
            self.proc = self.pool.apply_async(func=LCCCDataGenerator.get_batch_data,
                                              args=(batch_examples, tokenizer, self.speaker_ids))

    def _get_chatlog(self):
        with open(join(self.conf.data_model_dir, self.conf.train_data_path), "r", encoding="utf8") as fr:
            chat_log = json.load(fr)
        logger.info("remove white space in sen...")
        chat_log = [[re.sub("\s", "", sen) for sen in sens if len(re.sub("\s", "", sen)) > 0] for sens in chat_log]
        # remove some sen
        logger.info("remove long sens...")
        new_chat_log, num_long_sen = [], 0
        for sens in tqdm(chat_log, desc="remove long sens"):
            seq_len = 2  # CLS and SEP
            for idx, sen in enumerate(sens):
                seq_len += (len(sen) + 1)
                if seq_len > self.conf.max_len:
                    num_long_sen += 1
                    idx -= 1
                    break
            new_chat_log.append(sens[0:idx + 1])
        logger.warning("num long dialogue:{},\t{}%".format(num_long_sen, num_long_sen * 100 / len(new_chat_log)))
        new_chat_log = [sens for sens in tqdm(chat_log, desc="remove sens lower than two") if len(sens) > 1]
        random.shuffle(new_chat_log)
        return new_chat_log

    def _get_batch_examples(self):
        batch_examples = []
        for example in self.data_iter:
            batch_examples.append(example)
            if len(batch_examples) == self.conf.batch_size:
                return batch_examples
        return None

    @staticmethod
    def get_batch_data(batch_examples: List[List[str]], tokenizer: BertTokenizer, speaker_ids: List[int]):
        if batch_examples is None:
            return None
        input_ids, attention_mask, token_type_ids = [], [], []
        max_len = -1
        for sens in batch_examples:
            t_ipt_ids, t_attn_mask, t_token_type = [], [], []
            for idx, sen in enumerate(sens):
                speaker_id = speaker_ids[idx % 2]
                t_ipt_ids.append(speaker_id)
                tokens = tokenizer.tokenize(sen)
                t_ipt_ids.extend(tokenizer.convert_tokens_to_ids(tokens))
                t_token_type.extend([speaker_id] * (1 + len(tokens)))
            t_ipt_ids = [tokenizer.cls_token_id] + t_ipt_ids + [tokenizer.sep_token_id]
            t_token_type = [tokenizer.cls_token_id] + t_token_type + [t_token_type[-1]]
            t_attn_mask = [1] * len(t_ipt_ids)
            assert len(t_ipt_ids) == len(t_attn_mask)
            assert len(t_ipt_ids) == len(t_token_type)
            max_len = len(t_ipt_ids) if max_len < len(t_ipt_ids) else max_len
            input_ids.append(t_ipt_ids)
            attention_mask.append(t_attn_mask)
            token_type_ids.append(t_token_type)
        # padding
        for idx in range(len(input_ids)):
            pad_list = [tokenizer.pad_token_id] * (max_len - len(input_ids[idx]))
            input_ids[idx].extend(pad_list)
            attention_mask[idx].extend([0] * len(pad_list))
            token_type_ids[idx].extend(pad_list)
        return {"input_ids": torch.LongTensor(input_ids), "attention_mask": torch.LongTensor(attention_mask),
                "token_type_ids": torch.LongTensor(token_type_ids)}

    def __iter__(self):
        return self

    def __next__(self):
        # 获取数据
        if self.conf.use_multi_proc:
            ipt = self.proc.get()
            batch_examples = self._get_batch_examples()
            # print("len(batch_examples)",len(batch_examples))
            self.proc = self.pool.apply_async(func=LCCCDataGenerator.get_batch_data,
                                              args=(batch_examples, self.tokenizer, self.speaker_ids))
        else:
            batch_examples = self._get_batch_examples()
            ipt = LCCCDataGenerator.get_batch_data(batch_examples, self.tokenizer, self.speaker_ids)
        # 重置迭代器
        if ipt is None:
            random.shuffle(self.chat_log)
            self.data_iter = iter(self.chat_log)
            if self.conf.use_multi_proc:
                self.proc.get()
                batch_examples = self._get_batch_examples()
                self.proc = self.pool.apply_async(func=LCCCDataGenerator.get_batch_data,
                                                  args=(batch_examples, self.tokenizer, self.speaker_ids))
            raise StopIteration
        else:
            return ipt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conf = GPT2ChatbotConf()
    conf.max_len = 200
    conf.train_data_path = "LCCC-base_train.json"
    ##################
    data = LCCCDataGenerator(conf, tokenizer=BertTokenizer.from_pretrained(
        join(conf.data_model_dir, conf.pretrained_model_dir)))
    # device = torch.device("cuda:0")
    # model = GPT2LMHeadModel.from_pretrained(r"G:\Data\Cdial-GPT").to(device)
    # model.eval()
    # losses = []
    # start = time.time()
    # with torch.no_grad():
    for step, batch in tqdm(enumerate(data)):
        if step % 10000 == 0:
            print(step, data.steps)
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         batch["labels"] = batch["input_ids"]
    #         loss = model(**batch)[0]
    #         losses.append(float(loss.cpu().data))
    # end = time.time()
    # print("model:{},\tdata:{},\tavg loss:{},\ttime:{}s,\tuse_multi_proc".format(conf.pretrained_model_dir,
    #                                                                             conf.train_data_path,
    #                                                                             sum(losses) / len(losses), end - start),
    #       conf.use_multi_proc)
# model:G:\Data\Novel_GPT,	data:F:\谷歌下载目录\LCCC-base_valid.json,	avg loss:8.375751199531555,	time:88.97343420982361s,	use_multi_proc False
# model:G:\Data\Novel_GPT,	data:F:\谷歌下载目录\LCCC-base_valid.json,	avg loss:8.39735714111328,	time:87.07709693908691s,	use_multi_proc True
