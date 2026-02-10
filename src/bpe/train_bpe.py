import regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DATA_PATH = "./data"
SPECIAL_TOCKEN = "<|endoftext|>"
with open(f"{DATA_PATH}/TinyStoriesV2-GPT4-valid.txt") as f:
    text: str = f.read()
words: list[str] = [x.strip() for x in regex.findall(PAT, text)]

word_count: dict[tuple[bytes, ...], int] = {}
for word in words:
    key = tuple([x.encode('utf-8') for x in list(word)])
    cnt = word_count.get(key)
    if cnt:
        word_count[key] = cnt + 1
    else:
        word_count[key] = 1

vocab_list = [b'<|endoftext|>'] + [bytes([x]) for x in range(256)]


def max_pair(
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


            
                





    