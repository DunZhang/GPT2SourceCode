import json
from os.path import join
from typing import Dict


class GPT2Config():
    def __init__(self, vocab_size=20, n_positions=1024, hidden_size=10, d_intermediate=40,
                 drop_out=0.1, num_layers=1, n_head=2, layer_norm_epsilon=1e-5):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.hidden_size = hidden_size
        self.d_intermediate = hidden_size * 4
        self.drop_out = drop_out
        self.num_layers = num_layers
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon

    @staticmethod
    def get_name_table():
        return {
            "vocab_size": "vocab_size",
            "n_positions": "n_positions",
            "n_embd": "hidden_size",
            "attn_pdrop": "drop_out",
            "embd_pdrop": "drop_out",
            "n_layer": "num_layers",
            "n_head": "n_head",
            "layer_norm_epsilon": "layer_norm_epsilon",
        }

    @classmethod
    def from_pretrained(cls, model_dir: str):
        with open(join(model_dir, "config.json"), "r", encoding="utf8") as fr:
            config = json.load(fr)
        name_table = cls.get_name_table()
        new_conf = {}
        for k, v in config.items():
            if k in name_table:
                new_conf[name_table[k]] = v
        ###
        return cls(**new_conf)


if __name__ == "__main__":
    conf = GPT2Config.from_pretrained(r"D:\zdm")
    print(conf.__dict__)
