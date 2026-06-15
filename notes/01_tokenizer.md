# CS336 Assignment 1 - Part 1: Byte-Pair Encoding Tokenizer

## 1. 本部分目标

本部分实现一个 byte-pair BPE tokenizer, 主要包括：
- 理解 Unicode UTF-8 bytes 的关系
- 实现 BPE tokenizer 的 encode / decode
- 在 TinyStories 和 OpenWebText 上训练 tokenizer
- 统计 tokenizer 的压缩率和吞吐量

最终 tokenizer 的作用是：把原始文本转化成一串整数 tokens IDs, 作为语言模型训练数据

## 2. 为什么不用 Unicode code point 直接做 tokenizer？

Unicode code point 空间很大，直接拿字符编号做词表会导致词表非常稀疏。
所以使用 byte-level tokenizer：

文本 string
-> UTF-8 bytes
-> BPE tokens
-> token IDs

这样做的好处是：任意文本都可以表示为 0~255 的 byte 序列，不会出现 OOV
