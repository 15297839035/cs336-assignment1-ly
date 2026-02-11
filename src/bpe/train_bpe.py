import regex
from src.bpe.text_chunk import find_chunk_boundaries
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DATA_PATH = "./data"
SPECIAL_TOCKEN = "<|endoftext|>"
with open(f"{DATA_PATH}/TinyStoriesV2-GPT4-valid.txt") as f:
    text: str = f.read()

vocab_list: list[bytes] = [b'<|endoftext|>'] + [bytes([x]) for x in range(256)]

def text_count(
    text: str
)->dict[tuple[bytes, ...], int]:
    
    words: list[bytes] = [x.strip().encode('utf-8') for x in regex.findall(PAT, text)]
    word_count: dict[tuple[bytes, ...], int] = {}
    for word in words:
        key = tuple(word[i:i+1] for i in range(len(word)))
        cnt = word_count.get(key)
        if cnt:
            word_count[key] = cnt + 1
        else:
            word_count[key] = 1

    return word_count


def cal_max_pair(
    word_count: dict[tuple[bytes, ...], int]
)->tuple[bytes, bytes]:
    
    pair_freq: dict[tuple[bytes, bytes], int] = {}
    for byte_tuple in word_count:
        for i in range(len(byte_tuple)):
            if (i+1 < len(byte_tuple)):
                pair: tuple[bytes, bytes] = (byte_tuple[i], byte_tuple[i+1])
                freq = pair_freq.get(pair)
                if freq:
                    pair_freq[pair] = freq + 1
                else:
                    pair_freq[pair] = 1
    return max(*pair_freq, key=lambda pair: pair_freq[pair])


def merge(
        max_pair: tuple[bytes, bytes],
        word_count: dict[tuple[bytes, ...], int]
        ) -> dict[tuple[bytes, ...], int]:
    
    new_word_count: dict[tuple[bytes, ...], int] = {}
    for byte_tuple in word_count:
        byte_list: list[bytes] = []
        i = 0

        while (i + 1 < len(byte_tuple)):
            if max_pair != (byte_tuple[i], byte_tuple[i + 1]):
                byte_list.append(byte_tuple[i])
                i = i + 1
            else: # 合并 i 和 i + 1
                byte_list.append(byte_tuple[i] + byte_tuple[i + 1])
                i = i + 2
            
        if i == len(byte_tuple) - 1:
            byte_list.append(byte_tuple[i])

        new_word_count[tuple(byte_list)] = word_count[byte_tuple]

    return new_word_count

def text_pbe(
    text: str,
    vocab_size,
    vocab_list: list[bytes]
)-> list[tuple[bytes, bytes]]:
    
    merges: list[tuple[bytes, bytes]] = []

    word_count = text_count(text)
    while (len(vocab_list) < vocab_size):
        max_pair = cal_max_pair(word_count)
        merge(max_pair, word_count)
        vocab_list.append(max_pair[0] + max_pair[1])
        merges.append(max_pair)
    
    return merges


word_count = text_count(text)
print(word_count)
max_pair = cal_max_pair(word_count)
print("maxpair", max_pair)
print(merge(max_pair, word_count))
                





    