# Problem (train_bpe): BPE Tokenizer Training (15 points)

## Deliverable

Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer.

## Input Parameters

Your BPE training function should handle (at least) the following input parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_path` | `str` | Path to a text file with BPE tokenizer training data. |
| `vocab_size` | `int` | A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens). |
| `special_tokens` | `list[str]` | A list of strings to add to the vocabulary. These special tokens do not otherwise affect BPE training. |

## Return Values

Your BPE training function should return the resulting vocabulary and merges:

| Return | Type | Description |
|--------|------|-------------|
| `vocab` | `dict[int, bytes]` | The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes). |
| `merges` | `list[tuple[bytes, bytes]]` | A list of BPE merges produced from training. Each list item is a tuple of bytes (`<token1>`, `<token2>`), representing that `<token1>` was merged with `<token2>`. The merges should be ordered by order of creation. |

## Testing

To test your BPE training function against our provided tests, you will first need to implement the test adapter at `adapters.run_train_bpe`. Then, run:

```bash
uv run pytest tests/test_train_bpe.py
```

Your implementation should be able to pass all tests.

## Optional Optimization

Optionally (this could be a large time-investment), you can implement the key parts of your training method using some systems language, for instance:

- **C++**: Consider `cppyy` for Python integration
- **Rust**: Use PyO3 for bindings

If you do this, be aware of which operations require copying vs reading directly from Python memory, and make sure to:
- Leave build instructions, or
- Make sure it builds using only `pyproject.toml`

> **Note**: The GPT-2 regex is not well-supported in most regex engines and will be too slow in most that do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but the `regex` package in Python is, if anything, even faster.