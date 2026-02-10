import regex
from typing import List, Tuple, Dict
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DATA_PATH = "./data"
SPECIAL_TOCKEN = "<|endoftext|>"
with open(f"{DATA_PATH}/TinyStoriesV2-GPT4-valid.txt") as f:
    text: str = f.readline()
words: List[str] = [x.strip() for x in regex.findall(PAT, text)]

word_count: Dict[Tuple[bytes], int] = {}
for word in words:
    key = tuple([x.encode('utf-8') for x in list(word)])
    cnt = word_count.get(key)
    if cnt:
        word_count[key] = cnt + 1
    else:
        word_count[key] = 1

vocab_list = [b'<|endoftext|>'] + [bytes([x]) for x in range(256)]

pair_freq: Dict[Tuple[bytes, bytes], int]
for byte_tuple in word_count:
    for i in range(len(byte_tuple)):
        





    