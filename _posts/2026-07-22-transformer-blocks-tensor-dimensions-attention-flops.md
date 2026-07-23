---
layout: post
title: "Notes on Transformer Blocks, Tensor Dimensions and Attention FLOPs"
math: true
---

Transformers process a sequence of token representations using repeated blocks of attention, normalization, residual connections, and feed-forward layers. Although the architecture can look complicated, most of its tensor dimensions follow one simple rule: each block preserves the sequence length and model width.

This article develops the dimensions of the main Transformer operations, describes pre-normalization encoder and decoder blocks, and derives the commonly quoted attention cost

$$
4n^2d.
$$

## Input representation

Suppose a sequence contains \(n\) tokens. After token embedding and positional encoding, the sequence is represented by a matrix

$$
X \in \mathbb{R}^{n \times d_{\text{emb}}},
$$

where:

- \(n\) is the sequence length;
- \(d_{\text{emb}}\) is the embedding dimension, also often called the model dimension or hidden size.

Each row of \(X\) is the representation of one token. A Transformer block changes the content of these representations, but normally preserves the shape

$$
n \times d_{\text{emb}}.
$$

## Queries, keys, and values

Self-attention begins by projecting the input into queries, keys, and values:

$$
Q = XW_Q,
$$

$$
K = XW_K,
$$

$$
V = XW_V.
$$

The projection matrices have dimensions

$$
W_Q \in \mathbb{R}^{d_{\text{emb}} \times d_k},
$$

$$
W_K \in \mathbb{R}^{d_{\text{emb}} \times d_k},
$$

$$
W_V \in \mathbb{R}^{d_{\text{emb}} \times d_v}.
$$

Therefore,

$$
Q \in \mathbb{R}^{n \times d_k},
$$

$$
K \in \mathbb{R}^{n \times d_k},
$$

$$
V \in \mathbb{R}^{n \times d_v}.
$$

The query and key dimensions must match because attention computes the matrix product

$$
QK^\top.
$$

Since

$$
Q \in \mathbb{R}^{n \times d_k}
$$

and

$$
K^\top \in \mathbb{R}^{d_k \times n},
$$

the attention-score matrix has shape

$$
QK^\top \in \mathbb{R}^{n \times n}.
$$

Entry \((i,j)\) measures how strongly token \(i\) attends to token \(j\).

## Scaled dot-product attention

A general attention operation can be written as

$$
\operatorname{Attn}(X)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + B
\right)V,
$$

where

$$
B \in \mathbb{R}^{n \times n}
$$

is an optional bias or mask.

The matrix inside the softmax has shape

$$
n \times n.
$$

The softmax is applied independently to each row, so its output also has shape

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + B
\right)
\in
\mathbb{R}^{n \times n}.
$$

The final multiplication is

$$
AV.
$$

Because

$$
A \in \mathbb{R}^{n \times n}
$$

and

$$
V \in \mathbb{R}^{n \times d_v},
$$

the attention output has shape

$$
\operatorname{Attn}(X)
\in
\mathbb{R}^{n \times d_v}.
$$

An output projection is commonly used to map this result back to the model dimension:

$$
W_O \in \mathbb{R}^{d_v \times d_{\text{emb}}},
$$

$$
\operatorname{Attn}(X)W_O
\in
\mathbb{R}^{n \times d_{\text{emb}}}.
$$

This final shape is necessary for the residual addition with the original input.

## Multi-head attention

Instead of performing one attention operation with the full model width, multi-head attention divides the representation into \(h\) heads.

If the model dimension is \(d_{\text{emb}}\), a common choice is

$$
d_k = d_v = d_h = \frac{d_{\text{emb}}}{h}.
$$

For each head \(r\),

$$
Q_r,K_r,V_r
\in
\mathbb{R}^{n \times d_h}.
$$

Each head produces

$$
H_r
\in
\mathbb{R}^{n \times d_h}.
$$

The head outputs are concatenated:

$$
\operatorname{Concat}(H_1,\ldots,H_h)
\in
\mathbb{R}^{n \times hd_h}.
$$

Since

$$
hd_h=d_{\text{emb}},
$$

the concatenated output has shape

$$
n \times d_{\text{emb}}.
$$

A final output projection then mixes information across heads while preserving that shape.

## Feed-forward network

After attention, each token is processed independently by a feed-forward network. A standard two-layer feed-forward network is

$$
\operatorname{FF}(X)
=
\phi(XW_1+b_1)W_2+b_2,
$$

where \(\phi\) is an activation such as ReLU, GELU, or SiLU.

The first projection expands the feature dimension:

$$
W_1
\in
\mathbb{R}^{d_{\text{emb}} \times d_{\text{ff}}},
$$

$$
XW_1
\in
\mathbb{R}^{n \times d_{\text{ff}}}.
$$

The second projection maps back to the model dimension:

$$
W_2
\in
\mathbb{R}^{d_{\text{ff}} \times d_{\text{emb}}},
$$

$$
\operatorname{FF}(X)
\in
\mathbb{R}^{n \times d_{\text{emb}}}.
$$

A common design uses

$$
d_{\text{ff}} \approx 4d_{\text{emb}},
$$

although modern gated feed-forward networks may use different expansion ratios.

The feed-forward network does not mix token positions. It applies the same learned transformation to every row of the sequence matrix.

## Layer normalization

For one token vector

$$
x \in \mathbb{R}^{d_{\text{emb}}},
$$

LayerNorm computes

$$
\operatorname{LN}(x)
=
\frac{x-\mu(x)}
{\sqrt{\operatorname{Var}(x)+\epsilon}}
\odot \gamma+\beta.
$$

The mean and variance are computed across the feature dimension of that token:

$$
\mu(x)
=
\frac{1}{d_{\text{emb}}}
\sum_{j=1}^{d_{\text{emb}}}x_j.
$$

The scale and shift parameters are learned:

$$
\gamma,\beta
\in
\mathbb{R}^{d_{\text{emb}}}.
$$

They are commonly initialized as

$$
\gamma=\mathbf{1},
$$

$$
\beta=\mathbf{0}.
$$

The same \(\gamma\) and \(\beta\) are shared across all token positions, but each Transformer normalization layer has its own learned parameters.

For

$$
X\in\mathbb{R}^{n\times d_{\text{emb}}},
$$

LayerNorm preserves the shape:

$$
\operatorname{LN}(X)
\in
\mathbb{R}^{n\times d_{\text{emb}}}.
$$

### RMSNorm

RMSNorm omits mean subtraction and normalizes only by the root mean square:

$$
\operatorname{RMSNorm}(x)
=
\frac{x}
{\sqrt{
\frac{1}{d_{\text{emb}}}
\sum_{j=1}^{d_{\text{emb}}}x_j^2
+\epsilon
}}
\odot\gamma.
$$

Usually,

$$
\gamma\in\mathbb{R}^{d_{\text{emb}}},
$$

and there is no learned bias term. Like LayerNorm, RMSNorm preserves the input shape.

## Pre-norm Transformer encoder block

In a pre-normalization Transformer, normalization is applied before each major sublayer.

Let the input to encoder block \(\ell\) be

$$
X_\ell
\in
\mathbb{R}^{n\times d_{\text{emb}}}.
$$

The attention sublayer is

$$
U_\ell
=
X_\ell
+
\operatorname{MHA}
\left(
\operatorname{LN}_1(X_\ell)
\right).
$$

Both terms in the residual addition have shape

$$
n\times d_{\text{emb}},
$$

so

$$
U_\ell
\in
\mathbb{R}^{n\times d_{\text{emb}}}.
$$

The feed-forward sublayer is

$$
X_{\ell+1}
=
U_\ell
+
\operatorname{FF}
\left(
\operatorname{LN}_2(U_\ell)
\right).
$$

Again, both terms have shape

$$
n\times d_{\text{emb}},
$$

so the block output is

$$
X_{\ell+1}
\in
\mathbb{R}^{n\times d_{\text{emb}}}.
$$

An encoder stack repeats this block several times. The sequence length \(n\) and model width \(d_{\text{emb}}\) remain unchanged through the stack.

## Encoder-decoder Transformers

An encoder-decoder Transformer processes two different sequences.

The source sequence has \(n\) tokens:

$$
X_{\text{src}}
\in
\mathbb{R}^{n\times d_{\text{emb}}}.
$$

After the encoder stack, the encoder memory is

$$
E
\in
\mathbb{R}^{n\times d_{\text{emb}}}.
$$

The target-side decoder sequence has \(m\) tokens:

$$
X_{\text{tgt}}
\in
\mathbb{R}^{m\times d_{\text{emb}}}.
$$

The values \(n\) and \(m\) need not be equal because the source and target sequences can have different lengths.

During training, the decoder input is the ground-truth target sequence shifted right. For example, if the desired output is

```text
Je t'aime <eos>
```

the decoder may receive

```text
<bos> Je t'aime
```

and learn to predict the next token at every position.

During inference, the decoder starts with a beginning-of-sequence token and repeatedly appends its own predictions. The current number of generated tokens determines \(m\).

## Pre-norm Transformer decoder block

A full encoder-decoder block contains three main sublayers:

1. masked self-attention;
2. cross-attention;
3. a feed-forward network.

Let

$$
Y_\ell
\in
\mathbb{R}^{m\times d_{\text{emb}}}
$$

be the decoder hidden state entering block \(\ell\).

The masked self-attention step is

$$
S_\ell
=
Y_\ell
+
\operatorname{MaskedMHA}
\left(
\operatorname{LN}_1(Y_\ell)
\right).
$$

Its output has shape

$$
S_\ell
\in
\mathbb{R}^{m\times d_{\text{emb}}}.
$$

The cross-attention step is

$$
C_\ell
=
S_\ell
+
\operatorname{CrossAttn}
\left(
\operatorname{LN}_2(S_\ell),E
\right).
$$

The decoder provides the queries:

$$
Q
\in
\mathbb{R}^{m\times d_k}.
$$

The encoder memory provides the keys and values:

$$
K
\in
\mathbb{R}^{n\times d_k},
$$

$$
V
\in
\mathbb{R}^{n\times d_v}.
$$

Therefore, the cross-attention score matrix has shape

$$
QK^\top
\in
\mathbb{R}^{m\times n}.
$$

The cross-attention output has one row for each decoder position:

$$
\operatorname{CrossAttn}
\in
\mathbb{R}^{m\times d_v}.
$$

After its output projection, it has shape

$$
m\times d_{\text{emb}},
$$

which makes the residual addition valid.

The feed-forward step is

$$
Y_{\ell+1}
=
C_\ell
+
\operatorname{FF}
\left(
\operatorname{LN}_3(C_\ell)
\right).
$$

The final decoder block output is

$$
Y_{\ell+1}
\in
\mathbb{R}^{m\times d_{\text{emb}}}.
$$

## Encoder self-attention and masked decoder self-attention

Encoder self-attention allows every source token to attend to every source position. Its score matrix is fully available:

$$
QK^\top
\in
\mathbb{R}^{n\times n}.
$$

Masked decoder self-attention uses the same basic attention formula, but adds a causal mask. A token at position \(i\) may attend only to positions \(j\) satisfying

$$
j\leq i.
$$

One additive causal mask is

$$
B_{ij}
=
\begin{cases}
0, & j\leq i,\\
-\infty, & j>i.
\end{cases}
$$

After the mask is added and softmax is applied, all attention probabilities for future positions become zero.

This prevents the decoder from seeing future target tokens during training and enforces autoregressive generation.

## FLOPs for one attention operation

Consider

$$
\operatorname{Attn}(X)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+B
\right)V,
$$

with

$$
Q,K\in\mathbb{R}^{n\times d_k},
$$

$$
V\in\mathbb{R}^{n\times d_v},
$$

$$
B\in\mathbb{R}^{n\times n}.
$$

We count one scalar multiplication and one scalar addition as two FLOPs.

### Computing the score matrix

The product

$$
QK^\top
$$

multiplies an \(n\times d_k\) matrix by a \(d_k\times n\) matrix.

There are \(n^2\) output entries. Each entry is a dot product of length \(d_k\), requiring approximately \(d_k\) multiplications and \(d_k\) additions.

Therefore, the leading-order cost is

$$
2n^2d_k.
$$

The exact count, if a length-\(d_k\) dot product uses \(d_k\) multiplications and \(d_k-1\) additions, is

$$
n^2(2d_k-1).
$$

### Scaling, bias, and softmax

Scaling the \(n\times n\) score matrix costs approximately

$$
n^2
$$

operations.

Adding the bias or mask costs another

$$
n^2.
$$

A basic row-wise softmax performs exponentials, summations, and divisions over \(n^2\) elements. Its cost is proportional to

$$
n^2.
$$

These terms matter in an exact implementation-level count, but they are usually lower order than the matrix multiplications when \(d_k\) and \(d_v\) are large.

### Multiplying attention probabilities by values

Let

$$
A
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+B
\right)
\in
\mathbb{R}^{n\times n}.
$$

The output is

$$
AV,
$$

where

$$
A\in\mathbb{R}^{n\times n}
$$

and

$$
V\in\mathbb{R}^{n\times d_v}.
$$

There are \(nd_v\) output entries. Each is a length-\(n\) dot product. The leading-order cost is therefore

$$
2n^2d_v.
$$

The exact multiplication-and-addition count is

$$
nd_v(2n-1).
$$

### Total core-attention cost

Keeping only the two dominant matrix multiplications gives

$$
2n^2d_k+2n^2d_v.
$$

Therefore,

$$
\operatorname{FLOPs}_{\text{core attention}}
\approx
2n^2(d_k+d_v).
$$

If

$$
d_k=d_v=d,
$$

then

$$
\operatorname{FLOPs}_{\text{core attention}}
\approx
2n^2(d+d),
$$

which simplifies to

$$
\boxed{
4n^2d
}.
$$

The two halves have a direct interpretation:

$$
\underbrace{2n^2d}_{QK^\top}
+
\underbrace{2n^2d}_{AV}
=
\underbrace{4n^2d}_{\text{core attention}}.
$$

This is the origin of the widely used \(4n^2d\) estimate.

## Exact count under one simple convention

Suppose every scalar multiplication, addition, division, and exponential counts as one operation, and softmax is implemented without counting numerical-stability operations such as finding and subtracting the row maximum.

The components are:

$$
QK^\top:
\quad
n^2(2d_k-1),
$$

$$
\text{scaling}:
\quad
n^2,
$$

$$
\text{bias addition}:
\quad
n^2,
$$

$$
\text{softmax}:
\quad
3n^2-n,
$$

$$
AV:
\quad
nd_v(2n-1).
$$

Adding them gives

$$
F
=
n^2(2d_k-1)
+n^2
+n^2
+(3n^2-n)
+nd_v(2n-1).
$$

After simplification,

$$
\boxed{
F
=
2n^2(d_k+d_v)
+
4n^2
-
n(d_v+1)
}.
$$

When only the dominant terms are retained,

$$
F
\approx
2n^2(d_k+d_v).
$$

When

$$
d_k=d_v=d,
$$

this becomes

$$
F
\approx
4n^2d.
$$

The exact lower-order terms depend on the chosen FLOP convention and softmax implementation. For this reason, architecture-level analyses normally report the dominant matrix-multiplication cost.

## Multi-head attention still gives \(4n^2d_{\text{emb}}\)

Suppose there are \(h\) heads and each head has width

$$
d_h=\frac{d_{\text{emb}}}{h}.
$$

For one head, the core-attention cost is

$$
4n^2d_h.
$$

Across all \(h\) heads,

$$
h\cdot 4n^2d_h
=
4n^2hd_h.
$$

Since

$$
hd_h=d_{\text{emb}},
$$

the total is

$$
\boxed{
4n^2d_{\text{emb}}
}.
$$

Thus, splitting attention into more heads does not change the leading-order core-attention FLOPs when the total model width remains fixed.

## Projection costs are separate

The estimate

$$
4n^2d
$$

covers the two sequence-mixing matrix multiplications:

$$
QK^\top
$$

and

$$
AV.
$$

It does not include the linear projections that create \(Q\), \(K\), and \(V\), or the final output projection.

If all projections map between width \(d\) representations, then the four dense projections have a leading-order cost

$$
4\cdot 2nd^2
=
8nd^2.
$$

Therefore, a common leading-order estimate for a complete self-attention layer is

$$
\boxed{
8nd^2+4n^2d
}.
$$

The first term is the projection cost and grows linearly with sequence length. The second term is the core attention cost and grows quadratically with sequence length.

This distinction explains why attention becomes expensive for long sequences.

## Summary of important dimensions

For encoder self-attention:

$$
X
\in
\mathbb{R}^{n\times d_{\text{emb}}},
$$

$$
Q,K
\in
\mathbb{R}^{n\times d_k},
$$

$$
V
\in
\mathbb{R}^{n\times d_v},
$$

$$
QK^\top
\in
\mathbb{R}^{n\times n},
$$

$$
\operatorname{Attn}(X)
\in
\mathbb{R}^{n\times d_v},
$$

and after the output projection,

$$
\operatorname{MHA}(X)
\in
\mathbb{R}^{n\times d_{\text{emb}}}.
$$

For the feed-forward network:

$$
XW_1
\in
\mathbb{R}^{n\times d_{\text{ff}}},
$$

$$
\operatorname{FF}(X)
\in
\mathbb{R}^{n\times d_{\text{emb}}}.
$$

For encoder-decoder cross-attention:

$$
Q
\in
\mathbb{R}^{m\times d_k},
$$

$$
K
\in
\mathbb{R}^{n\times d_k},
$$

$$
V
\in
\mathbb{R}^{n\times d_v},
$$

$$
QK^\top
\in
\mathbb{R}^{m\times n},
$$

$$
\operatorname{CrossAttn}
\in
\mathbb{R}^{m\times d_v}.
$$

A pre-norm residual block preserves its input shape because every sublayer is projected back to the model dimension before the residual addition.

The central attention FLOPs result is

$$
\boxed{
2n^2d_k+2n^2d_v
}.
$$

When the key and value widths are both \(d\),

$$
\boxed{
4n^2d
}.
$$
