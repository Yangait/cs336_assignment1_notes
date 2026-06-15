# CS336 Assignment 1 - Part 1: Byte-Pair Encoding Tokenizer

## 1. 本部分目标

本部分实现一个 byte-pair BPE tokenizer, 主要包括：
- 理解 Unicode UTF-8 bytes 的关系
- 实现 BPE tokenizer 的 encode / decode
- 在 TinyStories 和 OpenWebText 上训练 tokenizer
- 统计 tokenizer 的压缩率和吞吐量

最终 tokenizer 的作用是：把原始文本转化成一串整数 tokens IDs, 作为语言模型训练数据
