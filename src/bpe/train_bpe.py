import regex
import os
import multiprocessing as mp
from typing import BinaryIO
import json
from src.bpe.text_chunk import find_chunk_boundaries

# GPT-4 tokenizer pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def text_count(
    text: str,
    word_count: dict[tuple[bytes, ...], int]
) -> None:

    words: list[bytes] = [x.encode('utf-8') for x in regex.findall(PAT, text)]
    for word in words:
        key = tuple(word[i:i+1] for i in range(len(word)))
        cnt = word_count.get(key)
        if cnt:
            word_count[key] = cnt + 1
        else:
            word_count[key] = 1


def cal_max_pair(
    word_count: dict[tuple[bytes, ...], int]
) -> tuple[bytes, bytes]:
    
    pair_freq: dict[tuple[bytes, bytes], int] = {}
    for byte_tuple in word_count:
        for i in range(len(byte_tuple)):
            if (i + 1 < len(byte_tuple)):
                pair: tuple[bytes, bytes] = (byte_tuple[i], byte_tuple[i+1])
                freq = pair_freq.get(pair)
                if freq:
                    pair_freq[pair] = freq + word_count[byte_tuple]
                else:
                    pair_freq[pair] = word_count[byte_tuple]
    return max(*pair_freq, key=lambda pair: (pair_freq[pair], pair[0], pair[1]))


def merge(
        max_pair: tuple[bytes, bytes],
        word_count: dict[tuple[bytes, ...], int]
        ) -> None:
    
    byte_tuples = tuple(word_count.keys())
    for byte_tuple in byte_tuples:
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

        new_byte_tuple = tuple(byte_list)
        if (new_byte_tuple != byte_tuple):
            
            word_count[new_byte_tuple] = word_count[byte_tuple]
            del word_count[byte_tuple]

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    word_count: dict[tuple[bytes, ...], int] = {}
    vocab_dict: dict[int, bytes] = {}
    for special_token in special_tokens:
        vocab_dict[len(vocab_dict)] = special_token.encode('utf-8')
    for i in range(256):
        vocab_dict[len(vocab_dict)] = bytes([i])
    merges_list: list[tuple[bytes, bytes]] = []

    pattern = '|'.join(map(regex.escape, special_tokens))

    with open(input_path) as f:
        for line in f:

            text = line
            text = regex.sub(pattern, '', text)
            text_count(text, word_count)

    while (len(vocab_dict) < vocab_size):
        max_pair = cal_max_pair(word_count)
        vocab_dict[len(vocab_dict)] = (max_pair[0] + max_pair[1])
        merges_list.append(max_pair)
        merge(max_pair, word_count)

    return vocab_dict, merges_list
                

def process_chunk(
    file_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """Process a single chunk of the file and return word counts."""
    word_count: dict[tuple[bytes, ...], int] = {}
    pattern = '|'.join(map(regex.escape, special_tokens))

    with open(file_path, 'rb') as f:
        f.seek(start)
        # Read until end position
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            text = line.decode('utf-8', errors='ignore')
            text = regex.sub(pattern, '', text)
            text_count(text, word_count)

    return word_count


def merge_word_counts(
    counts: list[dict[tuple[bytes, ...], int]]
) -> dict[tuple[bytes, ...], int]:
    """Merge multiple word count dictionaries."""
    merged: dict[tuple[bytes, ...], int] = {}
    for count_dict in counts:
        for key, value in count_dict.items():
            if key in merged:
                merged[key] += value
            else:
                merged[key] = value
    return merged


def train_bpe_process(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    n_processes: int = 4,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Multi-process BPE training."""
    # Initialize vocabulary with special tokens and base bytes
    vocab_dict: dict[int, bytes] = {}
    for special_token in special_tokens:
        vocab_dict[len(vocab_dict)] = special_token.encode('utf-8')
    for i in range(256):
        vocab_dict[len(vocab_dict)] = bytes([i])
    merges_list: list[tuple[bytes, bytes]] = []

    # Open file and find chunk boundaries
    split_token = special_tokens[0].encode('utf-8') if special_tokens else b'<|endoftext|>'

    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, n_processes, split_token)

    # Process chunks in parallel
    with mp.Pool(n_processes) as pool:
        tasks = []
        for i in range(len(boundaries) - 1):
            tasks.append((input_path, boundaries[i], boundaries[i + 1], special_tokens))

        results = pool.starmap(process_chunk, tasks)

    # Merge word counts from all processes
    word_count = merge_word_counts(results)

    # Perform BPE merges on merged counts
    while len(vocab_dict) < vocab_size:
        max_pair = cal_max_pair(word_count)
        vocab_dict[len(vocab_dict)] = (max_pair[0] + max_pair[1])
        merges_list.append(max_pair)
        merge(max_pair, word_count)

    return vocab_dict, merges_list


if __name__ == "__main__":

    DATA_PATH = "./data/TinyStoriesV2-GPT4-train.txt"
    SPECIAL_TOCKENS = ["<|endoftext|>"]

    VOCAB_PATH = "./src/bpe/target/TinyStoriesVocab.json"

    vocab_dict, _ = train_bpe_process(DATA_PATH, 10000, SPECIAL_TOCKENS, 10)
    
    with open(VOCAB_PATH, 'w') as f:
      vocab_str = {k: v.decode('utf-8', errors='replace') for k, v in vocab_dict.items()}
      f.write(json.dumps(vocab_str, indent=2))


    