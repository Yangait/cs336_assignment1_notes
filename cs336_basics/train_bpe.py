from collections import Counter, defaultdict   # Counter 统计频率，比如 pre-token 出现次数、pair 出现次数; defaultdict 自动创建默认值，比如 defaultdict(set)
import regex as re  # 注意不是 Python 自带 re，而是第三方 regex，因为要支持 \p{L}
import os  # 获取 CPU 核心数
from multiprocessing import Pool # 多进程并行处理文件块
import gc # 手动触发垃圾回收，释放大对象内存

BYTE_TOKENS = [bytes([i]) for i in range(256)]  # 准备 byte-level BPE 的基础单位
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""") #PAT 是 GPT-2 风格 pre-tokenizer，用来把文本粗切成 pre-token


# 按 special tokens 切开
def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split text by special tokens and remove the special tokens themselves.

    During BPE training, special tokens act as hard boundaries:
    merges should not cross them, and they should not contribute to
    merge statistics.
    """
    if not special_tokens:
        return [text]

    # Escape special regex characters and match longer tokens first.
    pattern = "|".join(
        re.escape(tok)
        for tok in sorted(special_tokens, key=len, reverse=True)
    )

    parts = re.split(pattern, text)

    # Remove empty strings caused by special tokens at the beginning/end
    # or consecutive special tokens.
    return [part for part in parts if part]

# 每一段用 regex pre-tokenize
def iter_pretokens(text: str):
    for match in re.finditer(PAT, text):
        yield match.group(0)
# 每个 pre-token 转成 UTF-8 bytes 再变为 tuple[bytes, ...]
def pretoken_to_byte_tuple(pretoken: str) -> tuple[bytes, ...]:
    raw = pretoken.encode("utf-8")
    return tuple(bytes([b]) for b in raw)

# 计数
def build_pretoken_counts(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    counts = Counter()
    parts = split_on_special_tokens(text, special_tokens)
    for part in parts:
        for pretoken in iter_pretokens(part):
            bt = pretoken_to_byte_tuple(pretoken)
            if bt:
                counts[bt] += 1
    return dict(counts)


# 计数
def update_pretoken_counts(counts: Counter, text: str, special_tokens: list[str]):
    parts = split_on_special_tokens(text, special_tokens)
    for part in parts:
        for pretoken in iter_pretokens(part):
            raw = pretoken.encode("utf-8")
            if raw:
                counts[tuple(BYTE_TOKENS[b] for b in raw)] += 1



def find_chunk_boundaries(
    input_path: str,
    num_chunks: int,
    boundary: bytes,
) -> list[int]:
    """
    Find safe chunk boundaries for parallel pre-tokenization.

    This version reads fixed-size blocks and also handles the case
    where `boundary` crosses two adjacent blocks.
    """
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")

    if not boundary:
        raise ValueError("boundary must be non-empty")

    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()

        if file_size == 0:
            return [0]

        chunk_size = file_size // num_chunks

        if chunk_size == 0:
            return [0, file_size]

        boundaries = [0]
        block_size = 4096

        for i in range(1, num_chunks):
            guess = i * chunk_size
            f.seek(guess)

            buffer = b""
            buffer_start = guess

            while True:
                block = f.read(block_size)

                if block == b"":
                    boundaries.append(file_size)
                    break

                buffer += block
                pos = buffer.find(boundary)

                if pos != -1:
                    boundaries.append(buffer_start + pos)
                    break

                if len(buffer) >= len(boundary):
                    keep = len(boundary) - 1
                    buffer_start = f.tell() - keep
                    buffer = buffer[-keep:]

        boundaries.append(file_size)

    return sorted(set(boundaries))

def build_pretoken_counts_from_chunk(args):
    input_path, start, end, special_tokens = args
    counts = Counter()

    with open(input_path, "rb") as f:
        f.seek(start)

        while f.tell() < end:
            pos_before = f.tell()
            line = f.readline()

            if not line:
                break

            pos_after = f.tell()

            if pos_after > end:
                keep = end - pos_before
                if keep <= 0:
                    break
                line = line[:keep]

            text = line.decode("utf-8", errors="replace")
            update_pretoken_counts(counts, text, special_tokens)

    return counts

# 多线程
def build_pretoken_counts_from_file_mp(
    input_path: str,
    special_tokens: list[str],
    num_processes: int | None = None,
    num_chunks: int | None = None,
):
    if num_processes is None:
        num_processes = os.cpu_count() or 1

    if num_chunks is None:
        num_chunks = num_processes * 8

    if num_processes <= 1 or not special_tokens:
        return build_pretoken_counts_from_file(input_path, special_tokens)

    boundary = special_tokens[0].encode("utf-8")
    boundaries = find_chunk_boundaries(input_path, num_chunks, boundary)

    jobs = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if start < end
    ]

    total_counts = Counter()

    with Pool(processes=num_processes) as pool:
        for counts in pool.imap_unordered(build_pretoken_counts_from_chunk, jobs, chunksize=1):
            total_counts.update(counts)

    return dict(total_counts)

# 单线程
def build_pretoken_counts_from_file(input_path, special_tokens):
     counts = Counter()

     with open(input_path, "r", encoding="utf-8", errors="replace", newline="") as f:
         for line in f:
             counts.update(build_pretoken_counts(line, special_tokens))

     return dict(counts)

# 词表初始化
def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    return vocab

# 统计一个 pre-token 里面所有相邻 pair
def get_pairs_in_seq(seq: tuple[bytes, ...]) -> Counter:
    pairs = Counter()
    for i in range(len(seq) - 1):
        pairs[(seq[i], seq[i + 1])] += 1
    return pairs

def init_pair_indexes(pretoken_counts: dict[tuple[bytes, ...], int]):
    seqs = list(pretoken_counts.keys())
    freqs = list(pretoken_counts.values())

    pair_counts = Counter()
    pair_to_seq_ids = defaultdict(set)

    for idx, seq in enumerate(seqs):
        freq = freqs[idx]
        pairs = get_pairs_in_seq(seq)

        for pair, n in pairs.items():
            pair_counts[pair] += freq * n
            # 说明 pair 来自哪个 idx
            pair_to_seq_ids[pair].add(idx)

    return seqs, freqs, pair_counts, pair_to_seq_ids


def remove_seq_from_indexes(
    idx: int,
    seq: tuple[bytes, ...],
    freq: int,
    pair_counts: Counter,
    pair_to_seq_ids: dict,
):
    old_pairs = get_pairs_in_seq(seq)

    for pair, n in old_pairs.items():
        pair_counts[pair] -= freq * n

        if pair in pair_to_seq_ids:
            pair_to_seq_ids[pair].discard(idx)

        if pair_counts[pair] <= 0:
            pair_counts.pop(pair, None)
            pair_to_seq_ids.pop(pair, None)
        elif pair in pair_to_seq_ids and not pair_to_seq_ids[pair]:
            pair_to_seq_ids.pop(pair, None)

def add_seq_to_indexes(
    idx: int,
    seq: tuple[bytes, ...],
    freq: int,
    pair_counts: Counter,
    pair_to_seq_ids: dict,
):
    new_pairs = get_pairs_in_seq(seq)

    for pair, n in new_pairs.items():
        pair_counts[pair] += freq * n
        pair_to_seq_ids[pair].add(idx)

def seq_contains_pair(seq: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> bool:
    left, right = pair
    for i in range(len(seq) - 1):
        if seq[i] == left and seq[i + 1] == right:
            return True
    return False



def select_best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:   
    return max(pair_counts.items(), key = lambda x: (x[1], x[0]))[0]


def merge_one_sequence(seq: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    left, right = pair
    merged = []
    i = 0

    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right:
            merged.append(left + right)
            i += 2
        else:
            merged.append(seq[i])
            i += 1
    return tuple(merged)



def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab = init_vocab(special_tokens)

    pretoken_counts = build_pretoken_counts_from_file_mp(
        input_path,
        special_tokens,
        num_processes=8,
    )

    seqs, freqs, pair_counts, pair_to_seq_ids = init_pair_indexes(pretoken_counts)
    del pretoken_counts  # 释放内存
    gc.collect()
    merges = []
    next_id = len(vocab)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = select_best_pair(pair_counts)
        affected_ids = list(pair_to_seq_ids.get(best_pair, set()))

        for idx in affected_ids:
            old_seq = seqs[idx]
            freq = freqs[idx]

            # 有时索引可能已经变化，保险起见检查一下
            if not seq_contains_pair(old_seq, best_pair):
                continue

            remove_seq_from_indexes(
                idx,
                old_seq,
                freq,
                pair_counts,
                pair_to_seq_ids,
            )

            new_seq = merge_one_sequence(old_seq, best_pair)
            seqs[idx] = new_seq

            add_seq_to_indexes(
                idx,
                new_seq,
                freq,
                pair_counts,
                pair_to_seq_ids,
            )

        new_token = best_pair[0] + best_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        merges.append(best_pair)

    return vocab, merges

