import torch
import torch.nn as nn
import torch.nn.functional as F
from GPT2.MultiHeadSelfAttention import MultiHeadSelfAttention
from GPT2Config import GPT2Config


class GPT2Layer(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2Layer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.self_attn = MultiHeadSelfAttention(d_ipt=config.hidden_size, n_head=config.n_head,
                                                dropout_p=config.drop_out)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.intermediate_linear1 = nn.Linear(config.hidden_size, config.d_intermediate, True)
        self.intermediate_linear2 = nn.Linear(config.d_intermediate, config.hidden_size, True)

        self.dropout = nn.Dropout(config.drop_out)
        self.dropout1 = nn.Dropout(config.drop_out)
        self.dropout2 = nn.Dropout(config.drop_out)

    def forward(self, src: torch.FloatTensor, src_mask: torch.FloatTensor) -> torch.FloatTensor:
        # multi head attention
        src1 = self.layer_norm1(src)
        src1 = self.self_attn(src1, src_mask)
        # add and norm
        src = src + self.dropout1(src1)

        # feed  forward
        src1 = self.layer_norm2(src)
        src1 = F.gelu(self.intermediate_linear1(src1))
        src1 = self.intermediate_linear2(src1)
        src1 = self.dropout(src1)
        # add and norm
        src = src + src1

        return src
