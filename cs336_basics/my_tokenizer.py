import regex as re
import pickle

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens = None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = special_tokens or []
        if self.special_tokens:
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.vocab.values():
                    new_id = max(self.vocab.keys()) + 1
                    self.vocab[new_id] = token_bytes
        # 反向表，token -> id  vocab : id -> token
        self.token_to_id = {token: idx for idx, token in self.vocab.items()}
        # 合并优先级
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}



    def decode(self, ids: list[int]) -> str:
        # pieces 为 一个 token 列表
        pieces = [self.vocab[idx] for idx in ids]
        # 变为普通字符串
        text = b"".join(pieces).decode("utf-8", errors = "replace")
        return text
    

    def _pretokenize(self, text: str):
        for match in PAT.finditer(text):
            yield match.group(0)

    def _bytes_to_initial_tokens(self, text: str):
        raw = text.encode("utf-8")
        return [bytes([b]) for b in raw]
    
    def _get_pairs(self, tokens: list[bytes]):
        pairs = []
        for i in range(len(tokens) - 1):
            pairs.append((tokens[i], tokens[i + 1]))
        return pairs
    
    def _merge_tokens(self, tokens: list[bytes], pair: tuple[bytes, bytes]):
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                tokens[i] = pair[0] + pair[1]
                del tokens[i + 1]
            else:
                i += 1
        return tokens
    
    def _bpe_encode_pretoken(self, pretoken: str):
        tokens = self._bytes_to_initial_tokens(pretoken)
        while True:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            best_pair = None
            best_rank = float("inf")
            for pair in pairs:
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_rank == float("inf"):
                break

            tokens = self._merge_tokens(tokens, best_pair)

        return tokens
    
    def _encode_ordinary_text(self, text: str):
        ids = []
        for pretoken in self._pretokenize(text):
            bpe_tokens = self._bpe_encode_pretoken(pretoken)
            for token in bpe_tokens:
                ids.append(self.token_to_id[token])
        return ids
    
    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_ordinary_text(text)

        ids = []
        special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens_sorted) + ")"
        parts = re.split(pattern, text)
        special_set = set(self.special_tokens)

        for part in parts:
            if part == "":
                continue

            if part in special_set:
                token_bytes = part.encode("utf-8")
                ids.append(self.token_to_id[token_bytes])
            else:
                ids.extend(self._encode_ordinary_text(part))

        return ids

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id


