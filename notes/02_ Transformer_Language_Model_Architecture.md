# CS336 Assignment 1 第三章学习笔记：Transformer LM Architecture

这章的核心目标是：**从零实现一个 decoder-only Transformer 语言模型**。输入是已经被 tokenizer 转成的 token id 序列，形状通常是 `(batch_size, sequence_length)`；模型输出每个位置预测“下一个 token”的 logits，形状是 `(batch_size, sequence_length, vocab_size)`。训练时用这些 logits 和真实下一个 token 计算 cross-entropy；生成时只取最后一个位置的分布，采样出下一个 token，再拼回输入继续生成。PDF 第 13 页的 Figure 1 展示了整体结构：Token Embedding → 多层 Transformer Block → Final Norm → Linear LM Head → Softmax/概率。

---

## 1. 第三章总体结构

第三章可以按下面这条主线理解：

```text
token_ids
  ↓
Token Embedding
  ↓
Transformer Block × num_layers
  ↓
Final RMSNorm
  ↓
Linear LM Head
  ↓
logits over vocabulary
```

每个 Transformer block 内部又分成两个子层：

```text
x
 ↓
RMSNorm
 ↓
Causal Multi-Head Self-Attention with RoPE
 ↓
Residual Add
 ↓
RMSNorm
 ↓
SwiGLU Feed-Forward Network
 ↓
Residual Add
 ↓
output
```

PDF 第 13 页 Figure 2 给出的就是这种 **pre-norm Transformer block**：先 Norm，再 Attention/FFN，最后加残差。

---

## 2. 语言模型到底在学什么？

语言模型的任务不是“理解一句话后给一个标签”，而是对序列中每一个位置都做 **next-token prediction**。

假设输入是：

```text
x1, x2, x3, ..., xm
```

模型在位置 `i` 输出一个向量 `oi`，它的长度等于词表大小 `vocab_size`。这个向量还不是概率，而是 **logits**。经过 softmax 后，每个维度表示下一个 token 是某个词表 token 的概率。

也就是说，第 `i` 个位置预测的是：

```text
p(x_{i+1} | x_1, x_2, ..., x_i)
```

一个 forward pass 会同时得到所有位置的预测，不需要一个位置一个位置跑模型。这也是 Transformer 训练效率高的原因之一。PDF 在后续 loss 部分也强调，Transformer 对每个位置计算 logits，再通过 softmax 得到下一个 token 的概率。

实现时要注意：**模型 forward 通常返回 logits，不要在模型内部直接 softmax**。因为训练时 cross-entropy loss 一般直接接收 logits，这样数值更稳定。

---

## 3. 张量形状总览

写 Transformer 最容易出错的地方不是公式，而是 **shape**。

建议先固定这些符号：

| 符号    | 含义                   | 常见变量名                        |
| ----- | -------------------- | ---------------------------- |
| `B`   | batch size           | `batch_size`                 |
| `T`   | 序列长度                 | `seq_len` / `context_length` |
| `D`   | 模型隐藏维度               | `d_model`                    |
| `V`   | 词表大小                 | `vocab_size`                 |
| `L`   | Transformer block 层数 | `num_layers`                 |
| `H`   | 注意力头数                | `num_heads`                  |
| `Dh`  | 每个 head 的维度          | `d_model // num_heads`       |
| `Dff` | FFN 中间层维度            | `d_ff`                       |

主干 shape 是：

```text
token_ids:        (B, T)
embedding output: (B, T, D)
block output:     (B, T, D)
LM head output:   (B, T, V)
```

CS336 特别强调：很多模块都应该支持额外的 batch-like dimensions，例如 attention heads 也可以看成 batch 维度；因此写函数时尽量让最后一维表示特征维，前面的维度都当作批处理维度。PDF 第 14 页说明了 batch、sequence length、attention heads 都可以被统一看成“批处理式维度”。

---

## 4. 数学记号和 PyTorch 记号的区别

数学里常写：

```text
y = W x
```

但 PyTorch 里一般把输入看成 row vector，所以实际实现通常是：

```text
y = x W^T
```

如果权重 `W` 存成 `(out_features, in_features)`，输入 `x` 是 `(..., in_features)`，那么输出应该是 `(..., out_features)`。因此 forward 里本质上做的是：

```text
x @ W.T
```

PDF 在 3.2.1 中专门提醒：作业数学推导主要用 column vector，但 PyTorch/NumPy 默认是 row-major，所以直接矩阵乘法时要注意转置；如果用 einsum，只要维度标签写清楚，就不容易混乱。

---

## 5. 参数初始化

这章不是随便初始化参数，而是给出了明确要求：

| 模块            | 初始化方式                                                |
| ------------- | ---------------------------------------------------- |
| Linear weight | 截断正态分布，均值 0，方差 `2 / (d_in + d_out)`，截断范围 `[-3σ, 3σ]` |
| Embedding     | 截断正态分布，均值 0，方差 1，截断范围 `[-3, 3]`                      |
| RMSNorm gain  | 全 1                                                  |

PDF 要求使用 `torch.nn.init.trunc_normal_` 初始化这些截断正态权重。

这一步的重要性在于：如果初始化太大，激活值和梯度可能爆炸；如果太小，信号可能传不动。虽然 pre-norm Transformer 对初始化相对鲁棒，但初始化仍然影响收敛速度和训练稳定性。

---

## 6. Linear Module

Linear 是 Transformer 中最基础的模块。第三章要求你自己实现 Linear，不能用 `nn.Linear` 或 `nn.functional.linear`。它没有 bias，符合现代 LLM 的常见做法。PDF 明确要求权重参数要存成 `W`，而不是 `W^T`，并放在 `nn.Parameter` 里。

核心理解：

```text
输入 x: (..., d_in)
权重 W: (d_out, d_in)
输出 y: (..., d_out)
```

数学写法：

```text
y = W x
```

PyTorch 实现思路：

```text
y = x @ W.T
```

容易错的点：

1. **权重 shape 不要写反**。
   应该是 `(out_features, in_features)`。

2. **不要加 bias**。
   这份作业的 Linear 没有 bias。

3. **forward 要支持任意前缀维度**。
   例如 `(B, T, D)` 输入也要能正常处理。

---

## 7. Embedding Module

Embedding 的作用是把离散 token id 映射成连续向量。输入是整数 token id，输出是每个 token 对应的向量表示。

```text
token_ids: (B, T)
embedding matrix: (V, D)
output: (B, T, D)
```

PDF 要求实现自定义 Embedding，不能使用 `nn.Embedding` 或 `nn.functional.embedding`；embedding matrix 要作为 `nn.Parameter`，并且 `d_model` 必须在最后一维。

直观理解：

```text
token id = 42
embedding[42] = 这个 token 的向量
```

所以 Embedding 本质上不是矩阵乘法，而是 **查表**。
它把原本没有大小关系的 token id 变成可以参与神经网络计算的 dense vector。

容易错的点：

1. token id 必须是整数张量，通常是 `torch.LongTensor`。
2. embedding matrix 的 shape 是 `(vocab_size, d_model)`。
3. 输出 shape 要保持输入的 batch 和 sequence 结构，只在最后多出 `d_model`。

---

## 8. Pre-Norm Transformer Block

原始 Transformer 是 post-norm：

```text
x -> sublayer -> residual add -> norm
```

但 CS336 这里实现的是现代 LLM 常用的 pre-norm：

```text
x -> norm -> sublayer -> residual add
```

一个完整 block 是：

```text
z = x + MultiHeadSelfAttention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

PDF 解释说，把 normalization 从子层输出之后移到子层输入之前，可以提升 Transformer 训练稳定性；pre-norm 的直观好处是从输入 embedding 到最终输出之间有一条更干净的 residual stream，有利于梯度流动。

这里的重点是：**残差连接不是可选装饰，而是主干路径**。Attention 和 FFN 都是在 residual stream 上做增量更新。

---

## 9. RMSNorm

RMSNorm 是这章的归一化模块。它不像 LayerNorm 那样减去均值，而是只根据均方根缩放激活值。

公式：

```text
RMS(a) = sqrt( mean(a_i^2) + eps )

RMSNorm(a_i) = a_i / RMS(a) * g_i
```

其中 `g_i` 是可学习的 gain 参数，长度是 `d_model`；`eps` 通常是 `1e-5`。PDF 还特别要求：在平方之前要把输入 upcast 到 `float32`，防止低精度下平方溢出，最后再 cast 回原来的 dtype。

输入输出 shape：

```text
input:  (B, T, D)
output: (B, T, D)
gain:   (D,)
```

RMSNorm 只在最后一维 `D` 上做归一化，不应该跨 batch 或 sequence 维度。

容易错的点：

1. **mean 是对最后一维求，不是对所有元素求**。
2. **gain 是可训练参数**，shape 是 `(d_model,)`。
3. **先转 float32 再算平方**，尤其是以后用 fp16/bfloat16 时很重要。
4. RMSNorm 不减均值，所以不要写成 LayerNorm。

---

## 10. Position-Wise Feed-Forward Network：SwiGLU

Transformer block 里的 FFN 是对每个位置独立做的非线性变换。它不会混合不同 token 的信息；不同 token 之间的信息交互主要靠 attention。

原始 Transformer 的 FFN 是：

```text
Linear -> ReLU -> Linear
```

而 CS336 采用现代 LLM 常用的 **SwiGLU**。PDF 说明，现代语言模型通常会使用更好的激活函数和 gating mechanism，例如 Llama 3、Qwen 2.5 使用的 SwiGLU。

SiLU 定义为：

```text
SiLU(x) = x * sigmoid(x)
```

SwiGLU 公式是：

```text
FFN(x) = W2( SiLU(W1 x) ⊙ W3 x )
```

其中：

```text
W1:  (d_ff, d_model)
W3:  (d_ff, d_model)
W2:  (d_model, d_ff)
```

CS336 要求 `d_ff` 近似取：

```text
d_ff ≈ 8/3 * d_model
```

并且通常要 round 到 64 的倍数，以更好利用硬件。

可以这样理解 SwiGLU：

```text
W1 x → gate branch
W3 x → value branch
SiLU(W1 x) 控制 W3 x 中哪些信息通过
W2 再投影回 d_model
```

也就是说，FFN 不只是“升维再降维”，它还通过门控机制选择性地保留或抑制信息。

---

## 11. RoPE：Rotary Position Embedding

Transformer 本身没有顺序感。也就是说，如果不加入位置信息，模型很难知道“第一个 token”和“第十个 token”的区别。

CS336 第三章使用的是 **RoPE**，即 Rotary Position Embedding。它不是像传统 absolute positional embedding 那样直接给 token embedding 加一个位置向量，而是在 attention 里面对 **query 和 key** 做旋转。PDF 说明：RoPE 会把 query/key 向量的相邻维度两两成对，根据 token 位置旋转一个角度。

对第 `i` 个位置、第 `k` 对维度，旋转角度是：

```text
θ_{i,k} = i / Θ^{(2k-2)/d}
```

每一对二维向量会乘一个旋转矩阵：

```text
[ cos θ   sin θ ]
[-sin θ   cos θ ]
```

RoPE 的几个关键点：

1. **只作用在 Q 和 K 上，不作用在 V 上**。
   因为 attention score 来自 `QK^T`，位置信息应该影响“谁关注谁”，而不是直接改 value 内容。

2. **RoPE 没有可学习参数**。
   cos/sin 可以提前算好，用 buffer 保存，而不是 `nn.Parameter`。PDF 建议可以用 `register_buffer(persistent=False)` 保存预计算的 sin/cos。

3. **head 维度当作 batch-like dimension**。
   多头注意力里每个 head 独立计算 attention，但同一位置的 RoPE 旋转规则相同。

4. **`d_k` 必须能两两配对**。
   因为 RoPE 是对相邻维度成对旋转，所以每个 head 的维度通常必须是偶数。

---

## 12. Scaled Dot-Product Attention

Attention 的核心问题是：对某个 query token，它应该从哪些 key/value token 中读取信息？

给定：

```text
Q: (..., Tq, d_k)
K: (..., Tk, d_k)
V: (..., Tk, d_v)
```

先计算注意力分数：

```text
scores = Q K^T / sqrt(d_k)
```

然后对最后一维做 softmax，得到每个 query 对所有 key 的注意力权重：

```text
attention_weights = softmax(scores)
```

最后用注意力权重加权求和 value：

```text
output = attention_weights V
```

输出 shape 是：

```text
(..., Tq, d_v)
```

为什么要除以 `sqrt(d_k)`？
因为 `QK^T` 是很多维度相乘再相加，`d_k` 越大，score 的方差越大。如果不缩放，softmax 容易变得过于尖锐，梯度不稳定。

---

## 13. Attention Mask

CS336 的 mask 规则很重要：

```text
True  = 允许 attend
False = 不允许 attend
```

PDF 明确说，mask 的 shape 是 `(seq_len, seq_len)`，每一行表示某个 query 可以看哪些 key；如果 mask 为 False，就应该在 softmax 之前把对应 score 加上 `-∞`，这样 softmax 后概率变成 0。

例如：

```text
[True, True, False]
```

表示当前 query 只能看前两个 key，不能看第三个 key。

注意：mask 要在 softmax 前加，不要在 softmax 后直接乘。
原因是 softmax 后乘 mask 会破坏概率归一化，除非你再重新归一化。

---

## 14. Causal Multi-Head Self-Attention

语言模型生成文本时，当前位置只能看过去，不能偷看未来。所以 decoder-only Transformer 必须使用 **causal mask**。

对序列：

```text
t1, t2, t3, ..., tn
```

第 `i` 个位置只能 attend 到：

```text
t1, t2, ..., ti
```

不能 attend 到：

```text
t_{i+1}, ..., tn
```

否则训练时模型会直接看到答案，next-token prediction 就被“泄题”了。PDF 第 25 页说明，causal masking 允许 token `i` attend 到所有 `j <= i` 的位置。

Multi-head attention 的公式是：

```text
MultiHeadSelfAttention(x)
= WO MultiHead(WQ x, WK x, WV x)
```

其中：

```text
WQ: (h * d_k, d_model)
WK: (h * d_k, d_model)
WV: (h * d_v, d_model)
WO: (d_model, h * d_v)
```

在本作业里通常：

```text
d_k = d_v = d_model / num_heads
```

所以 `h * d_k = d_model`，Q/K/V 投影后的总维度仍然是 `d_model`。PDF 也强调，Q、K、V 三个投影应该总共用三个矩阵乘法完成；更进一步的优化可以把它们合成一个大矩阵乘法。

典型 shape 流程：

```text
x:        (B, T, D)

Q/K/V:    (B, T, D)
reshape:  (B, T, H, Dh)
transpose:(B, H, T, Dh)

attention output per head:
          (B, H, T, Dh)

concat:   (B, T, D)
output projection:
          (B, T, D)
```

容易错的点：

1. causal mask 是下三角为 True。
2. RoPE 作用在 Q/K，且通常在 reshape 成多头之后应用。
3. attention 独立作用于每个 head。
4. concat heads 后要投影回 `d_model`。
5. 输出 shape 必须和输入 `x` 一样，方便残差相加。

---

## 15. Transformer Block 组装

一个 block 的结构可以写成：

```text
z = x + MHA(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

注意第二个 RMSNorm 的输入是 `z`，不是原来的 `x`。

从功能上看：

| 模块               | 作用                        |
| ---------------- | ------------------------- |
| RMSNorm          | 稳定激活尺度                    |
| Causal MHA       | 让当前位置聚合过去 token 的信息       |
| Residual Add     | 保留原信息，让模块学习增量             |
| SwiGLU FFN       | 对每个位置做非线性特征变换             |
| 第二个 Residual Add | 再次把变换结果加入 residual stream |

PDF 第 26 页要求实现 pre-norm Transformer block，并指出每个 block 有两个子层：multi-head self-attention 和 SwiGLU feed-forward；每个子层都是先 RMSNorm，再主操作，最后 residual connection。

---

## 16. Full Transformer LM 组装

完整模型需要这些参数：

```text
vocab_size
context_length
num_layers
d_model
num_heads
d_ff
rope_theta
```

整体 forward：

```text
token_ids
  → token embedding
  → block 1
  → block 2
  → ...
  → block L
  → final RMSNorm
  → LM head
  → logits
```

PDF 第 26 页说明，完整 Transformer LM 要先进行 embedding，然后送入 `num_layers` 个 Transformer blocks，最后经过 final layer norm 和 LM head 得到词表上的未归一化分布，也就是 logits。

注意：

1. **最后还有一个 RMSNorm**。
   因为这是 pre-norm 架构，block 内部的 norm 都在子层之前，所以所有 block 结束后还需要 final norm。

2. **LM head 输出 logits，不是 token id**。
   shape 是 `(B, T, V)`。

3. **训练时 targets 通常是输入右移一位**。
   例如输入 `[x1, x2, x3]`，目标是 `[x2, x3, x4]`。

---

## 17. 参数量估算

设：

```text
V = vocab_size
D = d_model
L = num_layers
Dff = d_ff
```

如果输入 embedding 和 LM head 不共享权重，则参数量大致为：

```text
Token embedding: V * D
LM head:         V * D

每个 Transformer block:
MHA:             4 * D^2
SwiGLU FFN:      3 * D * Dff
RMSNorm:         2 * D

Final RMSNorm:   D
```

所以总参数量：

```text
Total = 2VD + L(4D^2 + 3D Dff + 2D) + D
```

其中 RoPE 没有可学习参数，因为它的 sin/cos 是固定 buffer。PDF 的 resource accounting 部分也要求把 Transformer forward 里的矩阵乘法列出来，并用矩阵乘法 FLOPs 规则估算计算量。

---

## 18. FLOPs 估算思路

矩阵乘法的基本规则是：

```text
A: (m, n)
B: (n, p)

AB 的 FLOPs ≈ 2mnp
```

PDF 第 27 页给出的就是这个规则：一个输出元素需要大约 `n` 次乘法和 `n` 次加法，所以总共是 `2mnp`。

一个 Transformer block 的主要矩阵乘法包括：

```text
Q projection
K projection
V projection
QK^T
Attention weights @ V
Output projection WO
FFN W1
FFN W3
FFN W2
```

如果只粗略看矩阵乘法，单层 forward 的主要 FLOPs 大致是：

```text
MHA projections + output: 8 B T D^2
Attention scores + weighted sum: 4 B T^2 D
FFN: 6 B T D Dff
```

所以每层大概是：

```text
8BTD^2 + 4BT^2D + 6BTD Dff
```

最终 LM head 还需要：

```text
2 B T D V
```

这个估算很重要，因为它能解释为什么长上下文很贵：attention 有 `T^2` 项，序列长度翻倍，attention score 的计算和显存压力会显著增加。

---

## 19. 第三章实现顺序建议

建议按测试顺序实现，不要直接写完整 Transformer：

```text
1. Linear
2. Embedding
3. RMSNorm
4. SwiGLU
5. RoPE
6. Scaled Dot-Product Attention
7. Causal Multi-Head Self-Attention
8. Transformer Block
9. Transformer LM
10. Resource Accounting
```

每一步都先保证 shape 正确，再考虑效率。第三章最容易出现的问题基本都是 shape、mask、RoPE 和 residual connection 顺序问题。

---

## 20. 常见错误总结

| 错误                             | 后果                     |
| ------------------------------ | ---------------------- |
| Linear 权重 shape 写成 `(in, out)` | 测试权重加载失败或矩阵乘法错误        |
| Linear forward 忘记转置            | 输出 shape 不对            |
| Embedding 误用 one-hot 矩阵乘法      | 低效且容易写复杂               |
| RMSNorm 对整个 tensor 求 mean      | 归一化维度错误                |
| RMSNorm 没有 upcast 到 float32    | 低精度训练可能溢出              |
| SwiGLU 只写成 SiLU，没有 gate        | 结构不符合要求                |
| `d_ff` 没有 round 到 64 倍数        | 可能不符合配置期望              |
| RoPE 作用到 V                     | 位置编码用错位置               |
| RoPE 用 Parameter 存 sin/cos     | 错把固定位置编码当成可学习参数        |
| mask 中 True/False 理解反了         | attention 方向完全错        |
| causal mask 写成上三角 True         | 模型偷看未来                 |
| block 中先 residual 再 norm       | 写成 post-norm，不符合本章要求   |
| 模型 forward 返回 softmax 概率       | 训练 loss 数值不稳定，也不符合常见接口 |

---

## 21. 一句话总结第三章

第三章的本质是：**把 token id 先变成向量，再通过多层 pre-norm Transformer block 反复进行“上下文信息聚合 + 非线性特征变换”，最后把每个位置的隐藏状态投影回词表空间，用来预测下一个 token。**

最重要的三条线索是：

```text
Embedding 负责 token 身份
Attention 负责 token 之间的信息交流
FFN 负责每个位置内部的非线性变换
```

而 RMSNorm、Residual、RoPE、Causal Mask 分别保证：

```text
RMSNorm: 训练稳定
Residual: 信息和梯度流动
RoPE: 注入位置信息
Causal Mask: 防止偷看未来
```

