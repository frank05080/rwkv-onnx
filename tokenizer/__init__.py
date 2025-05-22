from .tokenizer import RWKV_TOKENIZER #, neox
import os

fname = "rwkv_vocab_v20230424.txt"
print("file is:", os.path.join(os.path.dirname(__file__), fname))
# world = RWKV_TOKENIZER(__file__[:__file__.rindex('/')] + '/' + fname)
world = RWKV_TOKENIZER(os.path.join(os.path.dirname(__file__), fname))
# neox = neox