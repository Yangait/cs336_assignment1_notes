# CS336 Assignment 1 Notes: Chapter 2 — Byte-Pair Encoding (BPE) Tokenizer

## 1. 本章目标

CS336 Assignment 1 第二章的目标是实现一个 **byte-level Byte-Pair Encoding tokenizer**。

语言模型不能直接处理字符串，例如：

```text
"hello world"
```

模型真正接收的是整数序列：

```text
[153, 421, 88, ...]
```

因此 tokenizer 的核心任务是：

```text
raw text  ->  token IDs
```

本章要完成两件事：

1. **训练 BPE tokenizer**

   * 输入：原始文本文件
   * 输出：`vocab` 和 `merges`

2. **实现 Tokenizer 类**

   * `encode`: text -> token IDs
   * `decode`: token IDs -> text
   * `encode_iterable`: 支持大文件流式编码

整章主线可以概括为：

```text
Unicode character
    ↓
Unicode code point
    ↓
UTF-8 bytes
    ↓
pre-tokenization
    ↓
BPE merges
    ↓
token bytes
    ↓
token IDs
    ↓
Transformer LM input
```

---

## 2. Unicode 与 Code Point

### 2.1 Unicode 是什么？

Unicode 是一个文本编码标准，它给每一个字符分配一个唯一的整数编号，这个编号叫做：

```text
Unicode code point
```

例如：

```python
ord("s")
# 115

ord("牛")
# 29275
```

所以：

```text
"s"  ->  115  ->  U+0073
"牛" -> 29275
```

反过来：

```python
chr(29275)
# "牛"
```

### 2.2 character、code point、encoding 的区别

这三个概念非常容易混淆：

| 概念         | 含义                        | 例子      |
| ---------- | ------------------------- | ------- |
| character  | 人看到的字符                    | `"牛"`   |
| code point | Unicode 给字符分配的整数编号        | `29275` |
| encoding   | 把 code point 存成 bytes 的规则 | UTF-8   |

所以：

```text
字符不是 byte
code point 也不是 byte
UTF-8 bytes 才是计算机实际存储和处理的字节序列
```

---

## 3. UTF-8 Encoding

### 3.1 为什么不直接用 Unicode code point？

如果直接在 Unicode code point 上训练 tokenizer，会遇到两个问题：

1. Unicode 字符数量很大，词表会很大。
2. 很多字符非常罕见，数据分布稀疏。

所以本章采用：

```text
Unicode string -> UTF-8 bytes
```

这样所有文本都可以表示为 `0~255` 范围内的 byte 值。

### 3.2 UTF-8 示例

```python
text = "hello! こんにちは!"
utf8_encoded = text.encode("utf-8")

print(utf8_encoded)
print(list(utf8_encoded))
```

可能得到：

```python
b'hello! \xe3\x81\x93\xe3\x82\x93...'
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, ...]
```

重点：

```text
一个 Unicode 字符不一定等于一个 byte
```

例如：

```python
len("hello! こんにちは!")
# 13

len("hello! こんにちは!".encode("utf-8"))
# 23
```

原因是英文字符通常占 1 byte，而日文、中文等字符通常占多个 bytes。

### 3.3 为什么选择 UTF-8？

相比 UTF-16 和 UTF-32，UTF-8 更适合作为 byte-level tokenizer 的底层编码：

1. UTF-8 是互联网文本中最常用的编码。
2. ASCII 英文字符在 UTF-8 中只占 1 byte。
3. UTF-32 对所有字符通常使用 4 bytes，空间浪费更大。
4. byte-level BPE 只需要 256 个基础 byte token，不会出现 OOV (UTF-16, UTF-32 也不会出现 OOV 因为都是 byte-level BPE. Word-level and Character-level tokenizer 会 OOV)。

OOV 指：

```text
out of vocabulary
```

也就是模型遇到词表中没有的 token。

byte-level tokenizer 的优点是：

```text
任何文本都可以被表示为 bytes，因此不会 OOV。
```

---

## 4. Subword Tokenization

### 4.1 三种 tokenization 粒度

常见 tokenizer 粒度有三种：

| 方法            | 优点                | 缺点             |
| ------------- | ----------------- | -------------- |
| word-level    | 序列短，语义直观          | 词表巨大，容易 OOV    |
| byte-level    | 不会 OOV，基础词表只有 256 | 序列太长，训练慢       |
| subword-level | 折中方案，常见片段合并       | 需要训练 tokenizer |

### 4.2 为什么 byte-level 还不够？

如果直接按 byte 切分，序列会很长。

例如：

```text
"the"
```

byte-level 会表示成：

```text
b"t", b"h", b"e"
```

这是 3 个 token。

但如果 `"the"` 在语料中非常常见，我们希望把它合并成一个 token：

```text
b"the"
```

这样可以减少序列长度，提高训练效率。

### 4.3 BPE 的核心思想

BPE，全称 Byte-Pair Encoding。

它的核心规则是：

```text
每次找到当前语料中最频繁的相邻 token pair，然后把它们合并成一个新 token。
```

例如：

```text
b"t", b"h" -> b"th"
b"th", b"e" -> b"the"
```

所以 BPE 本质上是一种压缩方法：

```text
用更大的 vocabulary 换取更短的 token 序列
```

也就是：

```text
vocab size 增大
sequence length 减小
```

---

## 5. BPE Tokenizer Training

BPE training 的目标是从训练语料中学习两个东西：

```python
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

其中：

* `vocab` 表示 token ID 到 token bytes 的映射。
* `merges` 表示 BPE merge 的顺序。

### 5.1 BPE training 三个步骤

本章的 BPE training 分为三步：

```text
1. Vocabulary initialization
2. Pre-tokenization
3. Compute BPE merges
```

---

## 6. Step 1: Vocabulary Initialization

因为本章实现的是 byte-level BPE，所以初始词表是所有可能的 byte：

```text
0, 1, 2, ..., 255
```

也就是 256 个基础 token。

每个基础 token 是一个 byte：

```python
bytes([0])
bytes([1])
...
bytes([255])
```

如果有 special token，例如：

```text
<|endoftext|>
```

也要加入 vocabulary。

因此最终词表大小大致为：

```text
final vocab size = 256 + number of special tokens + number of BPE merges
```

例如：

```text
vocab_size = 10000
special_tokens = ["<|endoftext|>"]
```

那么最多可以进行：

```text
10000 - 256 - 1
```

次 merge。

---

## 7. Step 2: Pre-tokenization

### 7.1 为什么需要 pre-tokenization？

pre-tokenization 是先把原文本按照某种规则分成多个细小部分(word-level), 然后对每个部分进行merge

理论上，可以直接对整个文本的 bytes 统计相邻 pair。

但这样有两个问题：

1. 效率很低
   每次 merge 都扫描整个 corpus，代价很大。

2. 容易产生不理想的 token
   例如：

```text
dog!
dog.
```

如果直接跨标点、空格合并，可能得到很多只差标点的 token。

因此需要先进行粗粒度切分，也就是 pre-tokenization。

### 7.2 pre-token 是什么？

pre-token 可以理解成：

```text
粗粒度切出来的一小段文本
```

例如：

```python
"some text that i'll pre-tokenize"
```

使用 GPT-2 风格 regex 可能切成：

```python
["some", " text", " that", " i", "'ll", " pre", "-", "tokenize"]
```

注意很多 pre-token 前面带空格：

```python
" text"
" that"
" i"
```

这是有意义的，因为空格本身也是文本的一部分。

### 7.3 为什么统计 pre-token 频率？

如果某个 pre-token 出现很多次，可以先统计它的频率。

例如：

```python
{
    "text": 10
}
```

那么在统计 pair 的时候：

```text
t e 出现 10 次
e x 出现 10 次
x t 出现 10 次
```

不需要在原始 corpus 中重复扫描 10 遍。

因此 BPE training 中常见的数据结构是：

```python
pretoken_counts = {
    (b"t", b"e", b"x", b"t"): 10,
    ...
}
```

注意：

```text
Python 没有单独的 byte 类型
单个 byte 也通常用 bytes 对象表示
```

例如：

```python
b"t"
```

---

## 8. Step 3: Compute BPE Merges

### 8.1 基本流程

BPE merge 的循环过程如下：

```text
while vocab is not full:
    1. 统计所有相邻 token pair 的频率
    2. 找到出现次数最多的 pair
    3. 如果频率相同，选择 lexicographically greater 的 pair
    4. 把这个 pair 合并成新 token
    5. 把新 token 加入 vocab
    6. 把 merge 记录加入 merges
```

例如：

```text
(b"t", b"h") -> b"th"
```

那么：

```python
merges.append((b"t", b"h"))
```

同时 vocabulary 中增加：

```python
b"th"
```

### 8.2 不能跨 pre-token 边界合并

BPE merge 只在每个 pre-token 内部进行。

例如 pre-token 是：

```python
["the", " cat"]
```

那么不能把 `"the"` 的最后一个 byte 和 `" cat"` 的第一个 byte 合并。

规则是：

```text
No merges across pre-token boundaries.
```

### 8.3 tie-breaking 规则

如果多个 pair 的频率相同，要选择字典序更大的 pair。

例如：

```python
max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
```

结果是：

```python
("BA", "A")
```

所以实现时不能随便取第一个最大值。

这一点非常重要，因为测试通常会检查 merge 顺序。

---

## 9. Special Tokens

### 9.1 special token 是什么？

special token 是具有特殊语义的字符串，例如：

```text
<|endoftext|>
```

它可以表示：

```text
文档结束
生成停止
不同 document 之间的边界
```

### 9.2 special token 的两个规则

训练 BPE 时：

```text
special token 是硬边界，不参与 merge 统计。
```

编码文本时：

```text
special token 必须作为一个完整 token 保留，不能被拆开。
```

例如：

```text
<|endoftext|>
```

不应该被切成：

```text
"<", "|", "end", "of", "text", "|", ">"
```

而应该作为一个整体 token ID。

### 9.3 为什么训练时要先移除 special token？

假设语料是：

```text
[Doc 1]<|endoftext|>[Doc 2]
```

训练时应该先按 special token 切开：

```text
[Doc 1]
[Doc 2]
```

然后分别进行 pre-tokenization。

这样可以避免：

```text
Doc 1 的结尾 和 Doc 2 的开头 被错误合并
```

---

## 10. BPE Training Example

假设训练语料是：

```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

pre-token 频率为：

```python
{
    "low": 5,
    "lower": 2,
    "widest": 3,
    "newest": 6
}
```

初始表示：

```text
low    -> l o w
lower  -> l o w e r
widest -> w i d e s t
newest -> n e w e s t
```

第一轮统计相邻 pair：

```text
lo, ow, we, er, wi, id, de, es, st, ne, ew
```

其中：

```text
es 和 st 都出现 9 次
```

频率相同，按字典序选择更大的：

```text
st
```

于是：

```text
s t -> st
```

然后继续合并：

```text
e st -> est
o w  -> ow
l ow -> low
w est -> west
n e -> ne
```

如果取 6 次 merge，`newest` 最终可能被编码为：

```text
[ne, west]
```

这个例子说明：

```text
BPE 不是按语义切词，而是按频率合并 token pair。
```

它可能学到类似词根、词缀的结构，但本质上是统计压缩。

---

## 11. BPE Tokenizer Encoding

训练结束后，我们得到了：

```python
vocab
merges
```

接下来要实现 tokenizer 的 `encode`。

### 11.1 encode 的过程

编码过程如下：

```text
1. 处理 special tokens
2. 对普通文本做 pre-tokenization
3. 把每个 pre-token 转成 UTF-8 bytes
4. 按 merges 的顺序应用 BPE merge
5. 把 token bytes 映射成 token IDs
```

### 11.2 训练时和编码时的区别

训练时：

```text
根据 corpus 统计 pair 频率，学习 merges。
```

编码时：

```text
使用已经学到的 merges，不再重新统计频率。
```

这是非常重要的区别。

### 11.3 Encoding example

假设输入是：

```text
the cat ate
```

pre-tokenizer 得到：

```python
["the", " cat", " ate"]
```

假设有 merges：

```python
(b"t", b"h")
(b"th", b"e")
```

则：

```text
[b"t", b"h", b"e"]
    ↓
[b"th", b"e"]
    ↓
[b"the"]
```

最后查表得到 token ID：

```text
"the" -> [9]
```

整体可能得到：

```python
[9, 7, 1, 5, 10, 3]
```

---

## 12. BPE Tokenizer Decoding

decode 的方向与 encode 相反。

流程是：

```text
token IDs
    ↓
token bytes
    ↓
concatenate bytes
    ↓
UTF-8 decode
    ↓
string
```

例如：

```python
ids = [9, 7, 1, 5, 10, 3]
```

查 vocab：

```text
9  -> b"the"
7  -> b" c"
1  -> b"a"
5  -> b"t"
10 -> b" at"
3  -> b"e"
```

拼接：

```python
b"the cat ate"
```

再 decode：

```python
b"the cat ate".decode("utf-8")
```

得到：

```text
the cat ate
```

### 12.1 为什么 decode 要用 errors="replace"？

并不是所有 token ID 序列都一定能组成合法 UTF-8。(UTF-8 的前缀会决定是多字节还是单字节)

如果出现非法 byte 序列，直接 decode 会报错。

因此应该使用：

```python
bytestring.decode("utf-8", errors="replace")
```

这样非法 byte 会被替换为：

```text
�
```

也就是 Unicode replacement character，U+FFFD。

---

## 13. Tokenizer 类接口

本章推荐实现如下接口：

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        pass

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def encode_iterable(self, iterable):
        pass

    def decode(self, ids: list[int]) -> str:
        pass
```

### 13.1 `__init__`

负责根据：

```python
vocab
merges
special_tokens
```

构造 tokenizer。

通常需要准备几个映射：

```python
id_to_token: dict[int, bytes]
token_to_id: dict[bytes, int]
merge_rank: dict[tuple[bytes, bytes], int]
```

其中 `merge_rank` 用来快速判断某个 pair 在 merges 中的优先级。

### 13.2 `from_files`

负责从文件中加载训练好的：

```python
vocab
merges
```

然后返回一个 tokenizer。

### 13.3 `encode`

负责：

```text
text -> token IDs
```

### 13.4 `encode_iterable`

负责大文件流式编码。

普通 `encode` 可能会一次性把整个文本读入内存，而大文件可能无法一次性加载。

所以需要：

```python
for token_id in tokenizer.encode_iterable(file):
    ...
```

目标是：

```text
memory-efficient tokenization
```

### 13.5 `decode`

负责：

```text
token IDs -> text
```

---

## 14. Experiments

本章还要求做 tokenizer 实验。

### 14.1 TinyStories tokenizer

要求：

```text
dataset: TinyStories
vocab_size: 10,000
special token: <|endoftext|>
```

需要记录：

```text
training time
memory usage
longest token
whether the longest token makes sense
```

TinyStories 是儿童故事数据集，文本相对简单，适合调试 tokenizer 和训练小模型。

### 14.2 OpenWebText tokenizer

要求：

```text
dataset: OpenWebText
vocab_size: 32,000
```

OpenWebText 更大、更复杂，更接近真实互联网文本。

对比 TinyStories tokenizer 和 OpenWebText tokenizer 时，可以从这些角度分析：

```text
1. vocabulary size
2. training corpus domain
3. longest token
4. compression ratio
5. behavior on out-of-domain text
```

一般来说：

```text
TinyStories tokenizer 更偏儿童故事和简单英文；
OpenWebText tokenizer 更偏互联网文本，覆盖范围更广。
```

### 14.3 Compression Ratio

压缩率通常用：

```text
bytes/token
```

计算：

```text
compression ratio = number of UTF-8 bytes / number of tokens
```

如果结果是：

```text
4 bytes/token
```

说明平均一个 token 表示 4 个 bytes。

压缩率越高，说明 token 序列越短。

但是 vocab 太大也会增加 embedding 和输出层的参数量，因此 tokenizer 设计存在 trade-off。

### 14.4 Throughput

throughput 表示 tokenizer 速度：

```text
bytes/second
```

计算方法：

```text
throughput = input bytes / tokenization time
```

如果测得 tokenizer 速度为：

```text
50 MB/s
```

那么处理 825GB 的 Pile 数据集，大约需要：

```text
825GB / 50MB/s
```

这类问题考察的是系统性能意识。

### 14.5 为什么保存 token IDs 用 uint16？

本章建议把 token IDs 保存成 NumPy 的 `uint16`。

原因是：

```text
uint16 可以表示 0 到 65535
```

而本章的 vocabulary size 通常是：

```text
TinyStories: 10,000
OpenWebText: 32,000
```

都小于 65536。

所以：

```text
uint16 足够表示所有 token ID，并且比 int32 / int64 更节省存储空间。
```

---

## 15. 实现路线图

我实现本章时，可以按以下顺序推进：

```text
1. 理解 Unicode / UTF-8 / bytes
2. 实现 pre-tokenization
3. 统计 pre-token 频率
4. 初始化 vocab：256 bytes + special tokens
5. 统计 pair counts
6. 实现一次 merge
7. 实现完整 BPE training loop
8. 返回 vocab 和 merges
9. 实现 Tokenizer.__init__
10. 实现 Tokenizer.encode
11. 实现 Tokenizer.decode
12. 实现 Tokenizer.encode_iterable
13. 跑 test_train_bpe.py
14. 跑 test_tokenizer.py
15. 训练 TinyStories tokenizer
16. 编码 TinyStories / OpenWebText 数据集为 .npy
```

---

## 16. 常见易错点

### 16.1 混淆 Unicode code point 和 UTF-8 bytes

错误理解：

```text
一个字符就是一个 byte
```

正确理解：

```text
一个字符有一个 Unicode code point；
这个 code point 用 UTF-8 编码后可能对应多个 bytes。
```

### 16.2 以为 byte-level token 永远是单 byte

初始 vocabulary 是 256 个 bytes，但 BPE merge 后，token 可以是多个 bytes：

```text
b"t" + b"h" -> b"th"
b"th" + b"e" -> b"the"
```

所以 byte-level BPE 的最终 token 是：

```text
byte sequence
```

不一定是单个 byte。

### 16.3 训练时和编码时混淆

训练时：

```text
根据频率学习 merges。
```

编码时：

```text
按照已有 merges 应用合并。
```

编码时不重新统计频率。

### 16.4 special token 处理错误

训练时：

```text
special token 是边界，不参与 merge 统计。
```

编码时：

```text
special token 必须作为整体 token 保留。
```

### 16.5 忘记 tie-breaking

pair 频率相同的时候，要选择：

```text
lexicographically greater pair
```

否则 merge 顺序可能和测试不一致。

### 16.6 跨 pre-token 边界 merge

BPE merge 不应该跨越：

```text
pre-token boundary
special token boundary
document boundary
```

---

## 17. 本章总结

第二章的核心是实现一个 byte-level BPE tokenizer。

它从 Unicode 字符串开始，先转成 UTF-8 bytes，再通过 BPE 学习常见 byte sequence，把文本压缩成更短的 token ID 序列。

最重要的链路是：

```text
character
    ↓
Unicode code point
    ↓
UTF-8 bytes
    ↓
pre-token
    ↓
BPE merges
    ↓
token bytes
    ↓
token ID
```

训练阶段学习：

```python
vocab
merges
```

编码阶段使用：

```python
vocab
merges
```

解码阶段执行：

```text
token IDs -> token bytes -> UTF-8 string
```

这部分是后续 Transformer LM 的输入基础。如果 tokenizer 错了，后面的模型训练也会受到影响。因此，BPE tokenizer 是整个语言模型训练 pipeline 的第一块核心组件。

````

你可以把文件命名为：

```text
notes/chapter2_bpe_tokenizer.md
````

或者放到你的项目 README 里作为：

```text
## Chapter 2: BPE Tokenizer
```
