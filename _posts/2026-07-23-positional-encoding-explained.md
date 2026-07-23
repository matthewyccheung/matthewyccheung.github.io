---
layout: post
title: "Notes on Transformer Positional Encodings"
math: true
---

Transformers are very good at comparing tokens with one another, but self-attention does not naturally know which token came first, which token came later, or how far apart two tokens are. If the same token vectors are presented in a different order, ordinary self-attention has no built-in mechanism that says the order changed.

Positional encoding solves this problem by giving the model information about token order. Different techniques place that information in different parts of the Transformer. Learned absolute embeddings and sinusoidal encodings are added to token embeddings before the sequence enters the Transformer layers. Rotary Position Embedding, usually called RoPE, is applied later, inside each attention layer, to the query and key vectors.

The difference is not merely an implementation detail. It changes how the model represents position and how naturally it can reason about relative distance.

## Why attention needs positional information

Suppose a sequence contains the tokens:

```text
the cat slept
```

Self-attention compares each token with every other token. It can learn that “cat” and “slept” are strongly related, but without positional information it does not inherently know whether “cat” came before “slept” or after it.

A token embedding represents properties of the token itself. A positional representation supplies information about where that token occurs in the sequence.

For token position 5, a model using an additive positional encoding receives something conceptually like:

```text
token embedding for "cat"
+
position representation for index 5
```

This produces a position-aware vector. The same word at position 20 receives a different combined representation.

## What embedding coordinates are

A Transformer represents each token as a vector. If the model dimension is 8, a token vector contains eight numbers:

```text
[x0, x1, x2, x3, x4, x5, x6, x7]
```

Each number is a coordinate, dimension, or feature of the embedding space. These terms refer to the individual entries of the vector.

A positional encoding has the same width as the token embedding so that the two vectors can be added. With an eight-dimensional model, the positional vector also contains eight coordinates.

The coordinates do not usually have simple human-readable meanings such as “plurality” or “past tense.” They are components of a distributed representation. Their importance comes from how the model combines them through matrix multiplication, dot products, and nonlinear transformations.

## Learned absolute positional embeddings

The simplest approach is to assign a trainable vector to every possible sequence position.

Let the token embedding at position n be:

$$
x_n
$$

Let the learned position vector for that position be:

$$
p_n
$$

The model input is:

$$
h_n = x_n + p_n
$$

Position 0 has one trainable vector, position 1 has another, and so on. During training, the model learns whatever positional patterns are useful for its task.

This method is called absolute because the vector identifies a specific location in the sequence. Position 7 always receives the position-7 vector.

The main limitation is that the model normally learns embeddings only for positions included in its configured context length. It also has no built-in rule saying that the relationship between positions 5 and 6 should resemble the relationship between positions 100 and 101. The model may learn such regularities, but the embedding table does not guarantee them.

## Sinusoidal positional encoding

Sinusoidal positional encoding also assigns a vector to every absolute token position, but the vector is not learned. It is generated from sine and cosine functions.

For a model dimension d, the coordinates are defined in pairs:

$$
PE(pos, 2i)
=
\sin\left(
\frac{pos}{10000^{2i/d}}
\right)
$$

$$
PE(pos, 2i+1)
=
\cos\left(
\frac{pos}{10000^{2i/d}}
\right)
$$

Here, `pos` is the token’s absolute position in the sequence. If the sequence begins at zero, the first token has `pos = 0`, the second has `pos = 1`, and so on.

The variable `i` is not the token position. It indexes a pair of embedding coordinates. The coordinate numbers themselves are `2i` and `2i+1`.

For an eight-dimensional positional encoding, the pairs are:

```text
i = 0  -> coordinates 0 and 1
i = 1  -> coordinates 2 and 3
i = 2  -> coordinates 4 and 5
i = 3  -> coordinates 6 and 7
```

Each pair uses one sine coordinate and one cosine coordinate with the same frequency.

## Why the coordinates change at different speeds

The denominator in the sinusoidal formula changes with `i`.

For an eight-dimensional example, the coordinate pairs use arguments resembling:

$$
pos
$$

$$
\frac{pos}{10}
$$

$$
\frac{pos}{100}
$$

$$
\frac{pos}{1000}
$$

The first pair therefore moves around its sine and cosine cycle quickly as the token position increases. The later pairs move more slowly.

“Changing quickly” means the numerical coordinate changes substantially when `pos` increases by one. “Changing slowly” means the coordinate changes only slightly over the same step.

This is similar to looking at several clock hands that rotate at different rates. A fast hand is good at distinguishing nearby moments, but it repeats frequently. A slow hand changes very little from one moment to the next, but it helps distinguish locations over a longer time range. Observing all the hands together gives a much richer description of time than observing only one.

Sinusoidal positional encoding uses the same idea. Fast-changing coordinates provide fine positional detail. Slow-changing coordinates provide broader location information. The full vector combines many scales.

## Are sinusoidal encodings absolute or relative?

Sinusoidal encodings are classified as absolute positional encodings because the vector is directly computed from the absolute position `pos`.

A token at position 5 receives:

$$
PE(5)
$$

A token at position 20 receives:

$$
PE(20)
$$

However, sinusoidal encodings have a useful algebraic structure that makes relative offsets easier to derive.

Consider one sine-cosine pair with angular frequency:

$$
\omega
$$

The positional pair at position n is:

$$
\begin{bmatrix}
\sin(n\omega) \\
\cos(n\omega)
\end{bmatrix}
$$

If the position is shifted by k, the new pair is:

$$
\begin{bmatrix}
\sin((n+k)\omega) \\
\cos((n+k)\omega)
\end{bmatrix}
$$

Using the angle-addition identities, this shifted pair can be obtained from the original pair with a fixed linear transformation:

$$
PE(n+k) = M_k PE(n)
$$

For one frequency pair, the matrix is:

$$
M_k =
\begin{bmatrix}
\cos(k\omega) & \sin(k\omega) \\
-\sin(k\omega) & \cos(k\omega)
\end{bmatrix}
$$

The important point is that this matrix depends on the offset k, not on the starting position n.

The same transformation represents “move forward by two positions” whether the move is from 5 to 7 or from 100 to 102.

This does not mean a standard attention layer automatically receives the relative position as an explicit input. The encoding remains absolute. It means the absolute vectors are organized so that fixed relative shifts correspond to consistent linear transformations.

That structure is more convenient for a neural network than a collection of unrelated position vectors because Transformers already rely heavily on learned linear maps.

This is not mainly a reduction in floating-point operations. The attention layer does not save work by explicitly evaluating a trigonometric identity. The benefit is representational: relative displacement has a regular linear structure that the model can exploit.

## Where sinusoidal encodings are placed

Sinusoidal encodings are normally added to token embeddings before the first Transformer layer:

$$
h_n = x_n + PE(n)
$$

The resulting vectors are then used to compute queries, keys, and values.

Conceptually, the flow is:

```text
token embeddings
+
sinusoidal position vectors
↓
Transformer layers
```

Because position is mixed into the residual stream at the beginning, every later layer can access positional information indirectly through its inputs.

## RoPE changes where position enters the model

Rotary Position Embedding takes a different approach. It does not usually add a position vector to the token embedding. Instead, it rotates the query and key vectors inside each attention layer.

The flow is:

```text
hidden states
↓
query, key, and value projections
↓
rotate queries and keys by position
↓
compute attention scores
```

The value vectors are usually left unchanged.

RoPE was introduced by Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu in the 2021 paper [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).

## How RoPE groups coordinates

Suppose one attention head produces an eight-dimensional query vector:

```text
[q0, q1, q2, q3, q4, q5, q6, q7]
```

RoPE groups the coordinates into two-dimensional pairs:

```text
(q0, q1)
(q2, q3)
(q4, q5)
(q6, q7)
```

Each pair is treated like a point in a two-dimensional plane. At token position `pos`, the pair is rotated by an angle determined by both the position and the pair’s frequency.

For pair index `i`, a common frequency definition is:

$$
\omega_i = 10000^{-2i/d}
$$

The angle used at position `pos` is:

$$
\theta_{pos,i} = pos \cdot \omega_i
$$

The first pair rotates quickly as position changes. Later pairs rotate more slowly. This is the same multi-frequency idea used by sinusoidal encoding, but applied as a transformation of queries and keys rather than as an added vector.

## What the RoPE rotation matrix looks like

For one pair of coordinates, the standard two-dimensional rotation matrix is:

$$
R(\theta)
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

Applied to a query pair, it gives:

$$
\begin{bmatrix}
q'_0 \\
q'_1
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
q_0 \\
q_1
\end{bmatrix}
$$

For a full vector, the conceptual matrix is block diagonal. With four coordinate pairs, it has the form:

$$
R_{pos}
=
\begin{bmatrix}
R(pos\omega_0) & 0 & 0 & 0 \\
0 & R(pos\omega_1) & 0 & 0 \\
0 & 0 & R(pos\omega_2) & 0 \\
0 & 0 & 0 & R(pos\omega_3)
\end{bmatrix}
$$

Expanded for an eight-dimensional vector, it looks like:

$$
R_{pos}
=
\begin{bmatrix}
c_0 & -s_0 & 0 & 0 & 0 & 0 & 0 & 0 \\
s_0 & c_0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & c_1 & -s_1 & 0 & 0 & 0 & 0 \\
0 & 0 & s_1 & c_1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & c_2 & -s_2 & 0 & 0 \\
0 & 0 & 0 & 0 & s_2 & c_2 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & c_3 & -s_3 \\
0 & 0 & 0 & 0 & 0 & 0 & s_3 & c_3
\end{bmatrix}
$$

where:

$$
c_i = \cos(pos\omega_i)
$$

$$
s_i = \sin(pos\omega_i)
$$

Implementations do not normally construct this large sparse matrix. They compute the equivalent coordinate-wise operations directly, which is much more efficient.

## Why RoPE produces relative position in attention

Let a query at position m be rotated by:

$$
R_m
$$

Let a key at position n be rotated by:

$$
R_n
$$

The attention score uses their dot product:

$$
(R_m q)^T(R_n k)
$$

This can be rearranged as:

$$
q^T R_m^T R_n k
$$

Rotation matrices have the property:

$$
R_m^T R_n = R_{n-m}
$$

Therefore:

$$
(R_m q)^T(R_n k)
=
q^T R_{n-m} k
$$

The positional part of the score depends on the difference between the two token positions:

$$
n-m
$$

This is the central idea behind RoPE. Each query and key is transformed using an absolute position, but when they interact through a dot product, the positional effect becomes relative.

That distinction is useful:

Sinusoidal encoding provides absolute position vectors whose structure makes relative offsets recoverable.

RoPE applies position-dependent rotations so that the query-key interaction itself naturally depends on relative displacement.

## Why rotate queries and keys but not values

Attention weights are produced from query-key similarity. Position needs to affect those similarity scores so the model can decide which tokens should attend to which other tokens.

The value vector contains the information that will be aggregated after the attention weights are computed. Rotating values is not necessary to create the relative-position effect in the attention score.

The usual calculation is therefore:

$$
q'_m = R_m q_m
$$

$$
k'_n = R_n k_n
$$

$$
v'_n = v_n
$$

The attention score is then based on:

$$
(q'_m)^T k'_n
$$

## The relationship between sinusoidal encoding and RoPE

Sinusoidal encoding and RoPE use closely related mathematical ingredients. Both rely on sine and cosine at multiple frequencies. Both organize coordinates into pairs. Both exploit the fact that shifting an angle corresponds to multiplication by a rotation matrix.

Their main difference is placement.

Sinusoidal encoding constructs a positional vector and adds it to the token representation before attention.

RoPE uses the same geometric idea to rotate queries and keys after their linear projections.

This placement gives RoPE a cleaner relationship to relative distance because query-key dot products reduce to a function of the positional difference.

## A compact mental model

Learned absolute embeddings act like a trainable address book. Every position receives its own learned address.

Sinusoidal embeddings act like a collection of clock hands rotating at different speeds. The joint state of the clocks identifies an absolute position, while the regular motion of the hands makes relative shifts mathematically structured.

RoPE takes those rotating clocks and applies them directly to the query and key coordinate pairs. When two rotated vectors are compared, only the difference between their rotation angles matters. That difference corresponds to the distance between the tokens.

## Final perspective

The main progression is from adding position to the input toward incorporating position directly into token-to-token interactions.

Learned absolute embeddings are simple and flexible but do not impose a regular relationship between positions.

Sinusoidal encodings are still absolute, but their multi-frequency structure gives relative shifts a consistent linear form.

RoPE goes further by turning position into rotations of queries and keys. Because the dot product between two rotated vectors depends on the difference in their angles, attention gains a direct and natural dependence on relative token position.

The result is a positional mechanism that fits the geometry of attention rather than merely attaching an address to each token.

## References

Vaswani, Ashish, et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.

Su, Jianlin, et al. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021.
