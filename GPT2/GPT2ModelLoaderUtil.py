import re
import torch
from collections import OrderedDict
from typing import Dict


class GPT2ModelLoaderUtil():
    @staticmethod
    def convert_huggingface_model(weights: Dict[str, torch.Tensor]):
        new_dict = OrderedDict()
        for name, weight in weights.items():
            if len(re.findall("h.\d+.attn.bias", name)) > 0:
                continue
            new_name, new_weight = GPT2ModelLoaderUtil.convert_name_weght(name, weight)
            new_dict[new_name] = new_weight
        return new_dict

    @staticmethod
    def convert_name_weght(name, weight: torch.Tensor):
        name = GPT2ModelLoaderUtil.convert_single_name(name)
        for s in ["self_attn.qkv_linear.weight", "self_attn.output_linear.weight",
                  "intermediate_linear1.weight", "intermediate_linear2.weight"]:
            if s in name:
                weight = weight.t()
                break
        return name, weight

    @staticmethod
    def convert_single_name(name: str):
        ss = name.split(".")
        if ss[0] == "transformer":
            ss = ss[1:]
        if ss[0] == "tokens_embed":
            ss[0] = "wte"
        if ss[0] == "positions_embed":
            ss[0] = "wpe"
        if ss[0] == "h":
            ss[0] = "layers"
        if ss[0] == "ln_f":
            ss[0] = "layer_norm"

        if len(ss) > 2 and re.match("ln_\d+", ss[2]):
            ss[2] = ss[2].replace("ln_", "layer_norm")
        if len(ss) > 2 and ss[2] == "attn":
            ss[2] = "self_attn"

        if len(ss) > 3 and ss[2] == "mlp":
            if ss[3] == "c_fc":
                ss = ss[0:3] + ss[4:]
                ss[2] = "intermediate_linear1"
            elif ss[3] == "c_proj":
                ss = ss[0:3] + ss[4:]
                ss[2] = "intermediate_linear2"

        if len(ss) > 3 and ss[3] == "c_attn":
            ss[3] = "qkv_linear"
        if len(ss) > 3 and ss[3] == "c_proj":
            ss[3] = "output_linear"
        return ".".join(ss)


if __name__ == "__main__":
    s = '''
    wte.weight
wpe.weight
h.0.ln_1.weight
h.0.ln_1.bias
h.0.attn.c_attn.weight
h.0.attn.c_attn.bias
h.0.attn.c_proj.weight
h.0.attn.c_proj.bias
h.0.ln_2.weight
h.0.ln_2.bias
h.0.mlp.c_fc.weight
h.0.mlp.c_fc.bias
h.0.mlp.c_proj.weight
h.0.mlp.c_proj.bias
ln_f.weight
ln_f.bias

    '''
    ss = [i.strip() for i in s.strip().split("\n")]

    for i in ss:
        print(GPT2ModelLoaderUtil.convert_single_name(i))
