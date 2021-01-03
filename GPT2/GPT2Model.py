import torch
import torch.nn as nn
from copy import deepcopy
from GPT2.GPT2Config import GPT2Config
from GPT2.GPT2Layer import GPT2Layer
from os.path import join
from GPT2ModelLoaderUtil import GPT2ModelLoaderUtil
import re


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.n_positions, config.hidden_size)
        self.drop = nn.Dropout(config.drop_out)
        self.layers = nn.ModuleList([deepcopy(GPT2Layer(config)) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 映射到词表的

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor = None,
                token_type_ids: torch.LongTensor = None, position_ids: torch.LongTensor = None, labels=None):
        inputs_embeds = self.wte(input_ids)
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], 1, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)  # TODO 为什么要用wte，不应该有专门的embedding吗
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)  # 16*128*768
        # print("after word embedding", hidden_states)
        # TODO 先生成一个大的对角矩阵，然后按需取就行了
        # bsz * seq_len * seq_len
        attn_mask = torch.tril(torch.ones(input_ids.shape[1], input_ids.shape[1]))
        attn_mask = attn_mask.to(input_ids.device)
        if attention_mask is not None:  # 除了考虑每个字只对前面的字可见，还要考虑，batch中的padding情况
            # 注意如果不乘以 attention_mask.unsqueeze(2)， 就和transformer2.1.1计算方式一致，但是理论上应该乘以的
            attn_mask = attn_mask.unsqueeze(0) * attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask)
        # print("after block",hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        # 计算loss
        if labels is None:
            return (hidden_states,)
        else:
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            return (loss, lm_logits, hidden_states)

    @classmethod
    def from_pretrained(cls, model_dir: str):
        # model def
        conf = GPT2Config.from_pretrained(model_dir)
        model = cls(config=conf)
        # get weight
        ori_dict = torch.load(join(model_dir, "pytorch_model.bin"))

        new_dict = GPT2ModelLoaderUtil.convert_huggingface_model(ori_dict)
        # for k, v in ori_dict.items():
        #     if len(re.findall("h.\d+.attn.bias", k)) > 0:
        #         continue
        #     # print(k, v.shape)
        #     print(k)
        # print("-" * 100)
        # for k in new_dict:
        #     print(k)
        # print("-" * 100)
        # for k, _ in model.named_parameters():
        #     print(k)
        model.load_state_dict(new_dict, strict=True)
        return model


if __name__ == "__main__":
    # model = GPT2Model.from_pretrained(r"D:\zdm")
    model = GPT2Model.from_pretrained(r"G:\PROJ_DATA_MODEL\nlpgeneration\Novel_GPT_ZD")
    model.eval()
    input_ids = torch.LongTensor([[1, 2, 3, 0],
                                  [4, 5, 6, 7],
                                  [9, 0, 0, 0]])
    attention_mask = torch.LongTensor([[1, 1, 1, 0],
                                       [1, 1, 1, 1],
                                       [1, 0, 0, 0]])
    logits = model(input_ids, attention_mask=None, labels=input_ids)[-1]
    print(logits.shape)
    print(logits)
