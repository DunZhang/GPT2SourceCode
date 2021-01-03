"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_ipt: int, n_head: int, dropout_p: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.qkv_linear = nn.Linear(d_ipt, d_ipt * 3, True)
        self.n_head = n_head
        self.output_linear = nn.Linear(d_ipt, d_ipt, True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src: torch.FloatTensor, attn_mask: torch.FloatTensor) -> torch.FloatTensor:
        # attn mask
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.dim()==3:
            attn_mask = attn_mask.unsqueeze(1)
        # generate q, k, v by Linear
        q, k, v = self.qkv_linear(src).chunk(3, dim=-1)  # bsz*seq_len*hid
        # change shape for multi head
        # q = q.contiguous().view(src.shape[0] * self.n_head, src.shape[1], src.shape[2] // self.n_head)
        # k = k.contiguous().view(src.shape[0] * self.n_head, src.shape[1], src.shape[2] // self.n_head)
        # v = v.contiguous().view(src.shape[0] * self.n_head, src.shape[1], src.shape[2] // self.n_head)
        q = q.contiguous().view(src.shape[0], src.shape[1], self.n_head, src.shape[2] // self.n_head).permute(0, 2, 1, 3) # bsz*n_head*seq_len*h
        k = k.contiguous().view(src.shape[0], src.shape[1], self.n_head, src.shape[2] // self.n_head).permute(0, 2, 3, 1) # bsz*n_head*h*seq_len
        v = v.contiguous().view(src.shape[0], src.shape[1], self.n_head, src.shape[2] // self.n_head).permute(0, 2, 1, 3) # bsz*n_head*seq_len*h
        # compute weight
        attn_weights = torch.matmul(q, k)  # bsz * n_head * seq_len * seq_len
        attn_weights = attn_weights * float((src.shape[2] // self.n_head)) ** -0.5
        attn_weights = attn_weights * attn_mask + (attn_mask - 1) * 1e4
        attn_weights = F.softmax(attn_weights, dim=-1)  # TODO 把dropout加上, attn_weights加
        attn_weights = self.dropout(attn_weights)
        # compute value
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(src.shape)
        attn_output = self.output_linear(attn_output)
        return attn_output
    # def forward(self, src: torch.FloatTensor, attn_mask: torch.FloatTensor) -> torch.FloatTensor:
    #     # attn mask
    #     if attn_mask.dim() == 2:
    #         attn_mask = attn_mask.unsqueeze(0)
    #     # generate q, k, v by Linear
    #     q, k, v = self.qkv_linear(src).chunk(3, dim=-1)  # bsz*seq_len*hid
    #     # change shape for multi head
    #     q = q.contiguous().view(src.shape[0] * self.n_head, src.shape[1], src.shape[2] // self.n_head)
    #     k = k.contiguous().view(src.shape[0] * self.n_head, src.shape[1], src.shape[2] // self.n_head).transpose(1, 2)
    #     v = v.contiguous().view(src.shape[0] * self.n_head, src.shape[1], src.shape[2] // self.n_head)
    #     # compute weight
    #     attn_weights = torch.bmm(q, k)  # (bsz*n_head) * seq_len * seq_len
    #     attn_weights = attn_weights * float((src.shape[2] // self.n_head)) ** -0.5
    #     attn_weights = attn_weights * attn_mask + (attn_mask - 1) * 1e4
    #     attn_weights = F.softmax(attn_weights, dim=-1)  # TODO 把dropout加上, attn_weights加
    #     attn_weights = self.dropout(attn_weights)
    #     # compute value
    #     attn_output = torch.bmm(attn_weights, v)
    #     attn_output = attn_output.contiguous().view(src.shape)
    #     attn_output = self.output_linear(attn_output)
    #     return attn_output


if __name__ == "__main__":
    d_ipt = 16
    n_head = 4
    dropout_p = 0.1
    src = torch.arange(0, 256, dtype=torch.float).view((8, 2, 16))  # bsz*seq_len*hid
    print(src.shape)
    src_mask = torch.zeros(size=(2, 2))
    model = MultiHeadSelfAttention(d_ipt, n_head, dropout_p)
    res = model(src, src_mask)
    print(res.shape)
