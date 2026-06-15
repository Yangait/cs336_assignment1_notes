# CS336 Assignment 1 学习记录：Building a Transformer LM from Scratch

本仓库用于记录我学习 Stanford CS336 Assignment 1: **Building a Transformer Language Model** 的过程。

这次作业的目标不是调用现成的 LLM 框架，而是从底层实现一个小型 decoder-only Transformer Language Model，并完整走通从文本数据到模型训练、验证、生成和实验分析的流程。

整体链路如下：

```text
raw text
→ byte-level BPE tokenizer
→ token ids
→ training batches
→ TransformerLM
→ cross-entropy loss
→ AdamW optimizer
→ training loop
→ checkpoint / evaluation
→ decoding / generation
→ TinyStories & OpenWebText experiments
```

---

## 1. Why This Repository?

我建立这个仓库的目的不是简单保存作业文件，而是系统复习和整理一次小型语言模型预训练的完整流程。

通过这次作业，我希望回答这些问题：

* 文本是如何被 tokenizer 转换成 token ids 的？
* byte-level BPE 为什么可以避免 OOV 问题？
* decoder-only Transformer LM 的每个模块具体在做什么？
* attention mask、RoPE、RMSNorm、SwiGLU 分别解决什么问题？
* language model 的训练目标为什么是 next-token prediction？
* AdamW 和普通 Adam 有什么区别？
* training loop 除了 forward/backward，还需要哪些工程组件？
* 生成文本时，temperature 和 top-p sampling 分别控制什么？
* TinyStories 和 OpenWebText 的训练难度为什么不同？
* 如何通过实验记录、learning curves 和 validation loss 分析模型表现？

---

## 2. Project Scope

本仓库记录的内容分为三个阶段。

### Phase 1: TinyStories Baseline

TinyStories 是一个简单的小故事数据集，适合快速调试 tokenizer、模型结构和训练循环。

本阶段目标：

* 训练 TinyStories byte-level BPE tokenizer
* 使用 10K vocabulary size
* 将 TinyStories train / validation 数据编码成 token ids
* 实现 decoder-only Transformer LM
* 实现 cross-entropy loss 和 AdamW optimizer
* 写完整 training loop
* 支持 checkpoint saving / loading
* 周期性评估 validation loss
* 实现 text generation
* 做 learning rate sweep
* 记录 train loss、validation loss、wall-clock time 和 generated samples

---

### Phase 2: OpenWebText Experiments

OpenWebText 更接近真实的 web-scale pretraining 数据，文本更复杂、主题更多样，噪声也更多。

本阶段目标：

* 训练 OpenWebText byte-level BPE tokenizer
* 使用 32K vocabulary size
* 比较 TinyStories tokenizer 和 OpenWebText tokenizer
* 比较两个 tokenizer 的 compression ratio
* 编码 OpenWebText train / validation 数据
* 使用相同或相近模型配置训练 OpenWebText LM
* 记录 OpenWebText learning curves
* 对比 TinyStories 和 OpenWebText validation loss
* 分析 OpenWebText 生成质量和 TinyStories 的差异
* 总结更真实数据集下训练小模型的困难

---

### Phase 3: Ablations and Improvements

在完成 baseline 后，进一步做一些消融实验和改进尝试。

计划包括：

* learning rate sweep
* batch size experiments
* context length experiments
* SwiGLU vs. SiLU
* different model sizes
* generation temperature / top-p comparison
* TinyStories vs. OpenWebText generation comparison
* training speed and bottleneck analysis
* validation loss and perplexity analysis

---

## 3. Repository Structure

计划中的仓库结构如下：

```text
cs336_assignment1_notes/
├── README.md
├── notes/
│   ├── 01_tokenizer.md
│   ├── 02_transformer_architecture.md
│   ├── 03_attention_and_rope.md
│   ├── 04_loss_adamw_training_loop.md
│   ├── 05_generation_decoding.md
│   ├── 06_tinystories_experiments.md
│   ├── 07_openwebtext_experiments.md
│   └── 08_debugging_and_takeaways.md
├── experiments/
│   ├── tinystories/
│   │   ├── lr_sweep.md
│   │   └── generated_samples.md
│   └── openwebtext/
│       ├── tokenizer_comparison.md
│       └── training_results.md
├── assets/
│   ├── loss_curves/
│   ├── diagrams/
│   └── generated_samples/
└── code_notes/
    ├── pseudocode_tokenizer.md
    ├── pseudocode_transformer.md
    └── pseudocode_generation.md
```

> Note: 本仓库主要用于学习记录、复习笔记和实验总结。完整可提交代码是否公开，需要根据课程要求和学术诚信要求谨慎处理。

---

## 4. Implementation Roadmap

### 4.1 Tokenizer

Tokenizer 部分主要复习：

* Unicode vs. UTF-8
* byte-level tokenization
* byte-level BPE
* pre-tokenization
* pair frequency counting
* merge rules
* special tokens
* encode / decode
* encode_iterable for large files

核心理解：

```text
Unicode string
→ UTF-8 bytes
→ byte-level BPE merges
→ token ids
```

容易出错的地方：

* special token 不能被拆开
* special token 在 BPE training 中应该作为 hard boundary
* BPE merge 不能跨 pre-token boundary
* pair frequency tie-breaking 要确定
* decode 时要处理非法 UTF-8 bytes
* 大文件编码不能一次性读入内存

---

### 4.2 Transformer Language Model

TransformerLM 的整体结构：

```text
token ids
→ token embedding
→ Transformer blocks
→ final RMSNorm
→ LM head
→ next-token logits
```

需要实现和复习的模块：

* Linear
* Embedding
* RMSNorm
* SwiGLU Feed-Forward Network
* Rotary Position Embedding, RoPE
* Scaled Dot-Product Attention
* Causal Multi-Head Self-Attention
* Transformer Block
* TransformerLM

关键 shape：

```text
input_ids:      (batch_size, sequence_length)
embeddings:     (batch_size, sequence_length, d_model)
q, k, v:        (batch_size, num_heads, sequence_length, head_dim)
attention:      (batch_size, num_heads, sequence_length, sequence_length)
hidden states:  (batch_size, sequence_length, d_model)
logits:         (batch_size, sequence_length, vocab_size)
```

最重要的理解：

```text
模型在每个位置预测下一个 token。
生成时只取最后一个位置的 logits。
```

---

### 4.3 Loss, Optimizer, and Training Loop

Language modeling 的训练目标是 next-token prediction。

如果原 token 序列是：

```text
[t1, t2, t3, ..., tn]
```

那么训练时：

```text
input:  [t1, t2, t3, ..., t(n-1)]
target: [t2, t3, t4, ..., tn]
```

训练流程：

```text
sample batch
→ forward pass
→ compute cross-entropy loss
→ backward pass
→ AdamW update
→ update learning rate
→ log train loss
→ evaluate validation loss
→ save checkpoint
```

需要记录的指标：

* step
* wall-clock time
* train loss
* validation loss
* learning rate
* tokens processed
* gradient norm
* generated samples

---

### 4.4 Generation / Decoding

文本生成流程：

```text
prompt
→ tokenizer.encode(prompt)
→ model(input_ids)
→ take logits[:, -1, :]
→ temperature scaling
→ softmax
→ top-p filtering
→ sample next token
→ append next token
→ repeat
```

需要支持：

* user-provided prompt
* max_new_tokens
* `<|endoftext|>` stopping
* temperature scaling
* top-p / nucleus sampling
* context_length truncation

关键点：

```text
temperature < 1: more conservative
temperature = 1: normal sampling
temperature > 1: more random
top-p: sample only from the smallest high-probability set whose cumulative probability reaches p
```

容易出错的地方：

* 生成时只取 `logits[:, -1, :]`
* temperature 是除法：`logits / temperature`
* top-p 之后要重新归一化
* 生成时要使用 `torch.no_grad()`
* prompt 或生成序列过长时，需要裁剪到 `context_length`

---

## 5. Experiments

### 5.1 TinyStories Experiments

基础配置计划：

| Hyperparameter |       Value |
| -------------- | ----------: |
| vocab size     |      10,000 |
| context length |         256 |
| d_model        |         512 |
| d_ff           |        1344 |
| num layers     |           4 |
| num heads      |          16 |
| RoPE theta     |       10000 |
| dataset        | TinyStories |

实验计划：

* baseline training
* learning rate sweep
* validation loss tracking
* generation samples
* overfit one minibatch debugging
* training speed profiling

Learning rate sweep table:

| Experiment | Learning Rate | Final Train Loss | Final Val Loss | Status | Notes         |
| ---------- | ------------: | ---------------: | -------------: | ------ | ------------- |
| TS-LR-1    |          1e-4 |              TBD |            TBD | TBD    | stable / slow |
| TS-LR-2    |          3e-4 |              TBD |            TBD | TBD    | baseline      |
| TS-LR-3    |          1e-3 |              TBD |            TBD | TBD    | faster        |
| TS-LR-4    |          3e-3 |              TBD |            TBD | TBD    | may diverge   |

---

### 5.2 OpenWebText Experiments

OpenWebText 实验计划：

| Item                 | Plan                                                   |
| -------------------- | ------------------------------------------------------ |
| tokenizer vocab size | 32,000                                                 |
| dataset              | OpenWebText                                            |
| comparison           | TinyStories tokenizer vs. OpenWebText tokenizer        |
| metrics              | compression ratio, validation loss, generation quality |
| goal                 | understand training difficulty on more realistic text  |

需要重点分析：

* OpenWebText tokenizer 学到的 token 是否更复杂？
* OpenWebText tokenizer 的 compression ratio 是否更高？
* TinyStories tokenizer 用在 OpenWebText 上会发生什么？
* 同样模型规模下，OpenWebText validation loss 为什么更高？
* OpenWebText 生成文本的 fluency 和 coherence 是否更差？
* 数据复杂度如何影响训练难度？

OpenWebText experiment table:

| Experiment    | Dataset     | Tokenizer       | Final Train Loss | Final Val Loss | Notes              |
| ------------- | ----------- | --------------- | ---------------: | -------------: | ------------------ |
| OWT-Tokenizer | OpenWebText | 32K BPE         |              N/A |            N/A | tokenizer training |
| OWT-Baseline  | OpenWebText | 32K BPE         |              TBD |            TBD | baseline LM        |
| OWT-Compare   | OpenWebText | TinyStories BPE |              TBD |            TBD | tokenizer mismatch |

---

## 6. Debugging Strategy

Transformer 实现很容易出现 silent bugs。我的调试策略是：

### 6.1 Overfit One Minibatch

先固定一个小 batch，不断训练同一批数据。

如果实现正确，training loss 应该能快速下降到接近 0。

如果不能 overfit 一个 batch，可能的问题包括：

* attention mask 写错
* target 没有右移
* optimizer 没有正确更新参数
* loss 维度处理错误
* RoPE shape 错误
* RMSNorm 数值不稳定
* causal attention 泄露未来信息

---

### 6.2 Shape Checking

重点检查：

```text
input_ids
embeddings
q / k / v
attention scores
attention mask
attention output
FFN output
logits
targets
loss
```

每个模块都要写清楚：

```text
input shape → output shape
```

---

### 6.3 Monitor Norms

训练时记录：

* weight norm
* activation norm
* gradient norm
* loss curve
* NaN / Inf

如果出现 loss 爆炸，优先检查：

* learning rate 是否太大
* softmax 是否数值稳定
* attention mask 是否正确
* AdamW 是否实现正确
* RMSNorm 是否 upcast 到 float32
* gradient norm 是否异常

---

## 7. Current Progress

| Part                    | Status      | Notes                                    |
| ----------------------- | ----------- | ---------------------------------------- |
| Unicode / UTF-8         | In progress | 理解 byte-level tokenizer 的基础              |
| BPE tokenizer           | In progress | TinyStories 10K, OWT 32K                 |
| Transformer modules     | In progress | Linear, Embedding, RMSNorm, SwiGLU, RoPE |
| Attention               | In progress | causal mask and multi-head attention     |
| Loss / AdamW            | In progress | from-scratch implementation              |
| Training loop           | In progress | logging, validation, checkpoint          |
| Generation              | In progress | temperature and top-p sampling           |
| TinyStories experiments | Not started | learning rate sweep                      |
| OpenWebText experiments | Not started | tokenizer and LM training                |
| Ablations               | Not started | model and training variations            |

---

## 8. Notes Index

Planned notes:

1. `01_tokenizer.md`
   Unicode, UTF-8, byte-level BPE, special tokens, encode/decode.

2. `02_transformer_architecture.md`
   TransformerLM overview, embeddings, residual stream, pre-norm blocks.

3. `03_attention_and_rope.md`
   scaled dot-product attention, causal mask, multi-head attention, RoPE.

4. `04_loss_adamw_training_loop.md`
   cross-entropy, AdamW, learning rate schedule, checkpointing.

5. `05_generation_decoding.md`
   decoding, temperature scaling, top-p sampling, context length.

6. `06_tinystories_experiments.md`
   TinyStories tokenizer, training curves, learning rate sweep, generated samples.

7. `07_openwebtext_experiments.md`
   OpenWebText tokenizer, compression ratio, training results, comparison with TinyStories.

8. `08_debugging_and_takeaways.md`
   bugs, debugging workflow, key lessons.

---

## 9. Key Takeaways

目前的核心理解：

1. Tokenizer 决定了原始文本如何进入模型。
2. byte-level BPE 同时解决了 OOV 和序列过长的问题。
3. Transformer 的实现难点主要在 tensor shape、causal mask 和 residual stream。
4. RoPE 给 attention 注入相对位置信息，但只作用在 q 和 k 上。
5. Language model 的训练目标是 next-token prediction。
6. 训练系统不仅包括模型本身，还包括 dataloading、logging、checkpoint、evaluation 和 generation。
7. TinyStories 适合快速调试，OpenWebText 更接近真实预训练场景。
8. 高质量实验记录应该同时包含配置、曲线、结果、失败案例和分析。

---

## 10. Final Goal

本仓库最终希望形成一份完整的 mini LLM pretraining 学习记录：

```text
from scratch implementation
+ conceptual explanation
+ debugging notes
+ TinyStories experiments
+ OpenWebText experiments
+ ablation studies
+ generation analysis
```

完成后，我希望自己不仅能“写出代码”，还能够清楚解释一个小型 Transformer LM 从数据、模型、训练到生成的完整过程。
