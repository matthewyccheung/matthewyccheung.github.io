---
layout: post
title:  "Transformers 0: A Simple Mental Model"
math: true
--- 

A transformer is a system that **routes information across tokens** and then **rewrites features within each token**, repeating this process many times. More formally, let the input be a sequence of tokens

\\[
X = \\begin{bmatrix} x_1^\\top \\\\ x_2^\\top \\\\ \\vdots \\\\ x_n^\\top \\end{bmatrix} \\in \\mathbb{R}^{n \\times d}, \\qquad x_i \\in \\mathbb{R}^d.
\\]

Here, \\(n\\) is the number of tokens and \\(d\\) is the feature dimension. You can think of \\(X\\) as the current **state** of the computation. Each row stores what the model currently knows about one token.

The cleanest view is that a transformer layer has two alternating operations:

1. **Attention:** decide which other tokens are relevant to each token and aggregate their information.
2. **MLP:** take the updated token representation and transform it locally in feature space.

So the transformer alternates between **communication across positions** and **computation within positions**.

A concise way to express this is:

\\[
\\text{Transformer layer} = \\text{token interaction} + \\text{feature update}.
\\]

That is the highest-level design principle behind almost every transformer variant.

## The attention mechanism as data-dependent routing

Given token matrix \\(X\\), the model forms three learned projections:

\\[
Q = X W_Q, \\qquad K = X W_K, \\qquad V = X W_V.
\\]

Here:

- \\(Q\\) is the **Query** matrix.
- \\(K\\) is the **Key** matrix.
- \\(V\\) is the **Value** matrix.

A good mental model is:

- **Query \\((Q)\\):** what a token is looking for.
- **Key \\((K)\\):** what a token offers to others as an address or identifier.
- **Value \\((V)\\):** the actual content that gets passed along if that token is attended to.

So for token \\(i\\), its query asks: *which other tokens are relevant to me right now?* Each other token \\(j\\) presents a key saying: *this is the kind of information I contain.* If the query and key match strongly, the model pulls in that token's value.

The similarity scores are computed using dot products:

\\[
S = \\frac{Q K^\\top}{\\sqrt{d_k}}.
\\]

The \\(\\sqrt{d_k}\\) scaling keeps the scores numerically stable as dimensionality grows. These scores are then normalized row-wise with softmax to obtain attention weights:

\\[
A = \\mathrm{softmax}\\!\\left( \\frac{Q K^\\top}{\\sqrt{d_k}} + B \\right).
\\]

Here \\(B\\) may include masks or positional bias terms. The final attention output is the weighted sum of values:

\\[
\\mathrm{Attn}(X) = A V = \\mathrm{softmax}\\!\\left( \\frac{Q K^\\top}{\\sqrt{d_k}} + B \\right) V.
\\]

Attention is often described as a **programmable kernel smoother** or **data-dependent routing rule**: each token becomes a weighted average of other tokens' value vectors, where the weights are computed dynamically from the current input.

## Why Q, K, and V are separated

The separation into query, key, and value is not just notation. It is a design choice that decouples three roles:

- the criterion used to decide relevance,
- the address used to expose what a token can provide,
- and the content actually transmitted.

If we used the same representation for all three roles, the model would be less flexible. With separate learned projections, the transformer can learn one subspace for matching and another for transporting content.

A useful expert summary is:

\\[
\\text{attention} = \\text{who should talk to whom} + \\text{what content should be sent}.
\\]

The \\(Q\\) and \\(K\\) terms determine the first part, while \\(V\\) determines the second.

## Multi-head attention as parallel relation discovery

In practice, transformers use multiple attention heads. If head \\(h\\) has projections \\(W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}\\), then each head computes

\\[
\\mathrm{Attn}^{(h)}(X) = \\mathrm{softmax}\\!\\left( \\frac{Q^{(h)} (K^{(h)})^\\top}{\\sqrt{d_k}} + B^{(h)} \\right) V^{(h)}.
\\]

The outputs are concatenated and projected back:

\\[
\\mathrm{MHA}(X) = \\mathrm{Concat}\\!\\big( \\mathrm{Attn}^{(1)}(X), \\dots, \\mathrm{Attn}^{(H)}(X) \\big) W_O.
\\]

Multi-headed attention allows the model to represent **different relation types in parallel**. One head may focus on local structure, another on long-range dependencies, another on positional alignment, and another on semantic correspondence. Whether the heads become interpretable depends on training, but architecturally that is why they exist.

## Residual structure: preserve state while refining it

Transformers do not replace the current representation outright. They **add corrections** through residual connections. In a simplified form, one can write:

\\[
X' = X + \\mathrm{Attn}(X) + \\mathrm{MLP}\\big( X + \\mathrm{Attn}(X) \\big).
\\]

This emphasizes the core idea that the model starts from the current state \\(X\\), adds an attention-based update, and then adds a nonlinear feature update through the MLP.

In modern implementations, especially the pre-normalization variant, this is typically written in two sequential steps:

\\[
X_{\\mathrm{attn}} = X + \\mathrm{Attn}(\\mathrm{LN}(X)).
\\]

\\[
X' = X_{\\mathrm{attn}} + \\mathrm{MLP}(\\mathrm{LN}(X_{\\mathrm{attn}})).
\\]

Residual connections make the transformer an **iterative refinement process** rather than a sequence of destructive rewrites. Each layer adjusts the state while preserving a stable information pathway through depth.

## The role of the MLP: local feature rewriting

After attention mixes information across tokens, the MLP transforms each token independently. A standard MLP block has the form:

\\[
\\mathrm{MLP}(x) = W_2 \\, \\phi(W_1 x + b_1) + b_2,
\\]

where \\(\\phi\\) is typically a nonlinear activation such as GELU or ReLU.

Applied token-wise to the matrix \\(X\\), the MLP does not communicate across positions. Instead, it performs a **channel-wise nonlinear rewrite** of the features at each token. Attention decides what information arrives; the MLP decides how that information is reorganized and amplified inside the token representation.

This leads to a very useful slogan:

\\[
\\text{Attention mixes tokens; the MLP mixes channels.}
\\]

This single sentence captures one of the deepest structural design choices in a transformer.

## LayerNorm and why pre-LN matters

\\(\\mathrm{LN}\\) is Layer Normalization. For a token feature vector \\(x \\in \\mathbb{R}^d\\), LayerNorm computes:

\\[
\\mathrm{LN}(x) = \\gamma \\odot \\frac{x - \\mu(x)}{\\sigma(x)} + \\beta,
\\]

where \\(\\mu(x)\\) and \\(\\sigma(x)\\) are the mean and standard deviation over the feature coordinates of that token, and \\(\\gamma\\) and \\(\\beta\\) are learnable parameters.

If we write this in component-agnostic form for a token matrix \\(X\\), the normalization is still applied independently to each token across its feature dimension. The purpose is to stabilize the scale of activations and improve optimization.

When people write \\(\\mathrm{Attn}(\\mathrm{LN}(X))\\) and \\(\\mathrm{MLP}(\\mathrm{LN}(X))\\), they are using **pre-normalization** or **Pre-LN**. The update becomes:

\\[
X_{\\mathrm{attn}} = X + \\mathrm{Attn}(\\mathrm{LN}(X)),
\\]

\\[
X' = X_{\\mathrm{attn}} + \\mathrm{MLP}(\\mathrm{LN}(X_{\\mathrm{attn}})).
\\]

Pre-LN makes the optimization problem better conditioned. In deep transformers, that often improves gradient flow and training stability compared with post-normalization designs.

## Transformer Design Choices

Transformer design can be few core design axes.

1. **Tokenization**: Before attention can operate, the model must decide what the tokens are. In language, tokens may be subwords. In vision, they may be image patches. In multimodal settings, they may include text tokens, image patches, or latent units. A deep way to think about tokenization is: **tokenization determines the state space of computation**. If the tokenization is too coarse, the model may miss important fine detail. If it is too fine, the sequence becomes very long and attention becomes expensive.

2. **Positional structure**: Attention by itself is permutation-equivariant over the token set. To encode order or geometry, one adds positional information or bias terms. This can be done with absolute positional embeddings, relative biases, rotary embeddings, or modality-specific structures. The key question is: **what symmetries should the model preserve, and what geometry should it know?** In language, order matters. In images, two-dimensional geometry matters. In medical imaging, three-dimensional or temporal structure may matter.

3. **Heads and projections**: The choice of number of heads, head dimension, and projection matrices determines the diversity of relations the model can represent in parallel. Too few heads may bottleneck relational structure; too many may be wasteful or unstable depending on scale.

4. **Depth and width**: A transformer layer is one round of routing plus rewriting. Stacking layers means repeating this process, so depth controls how many iterative refinement steps the model can apply. Width controls the capacity of each token representation. A good mental model is: depth \\approx \\text{number of computational refinement steps and width \\approx size of the working memory per token.

5. **Residual and normalization design**: Residual pathways and normalization are key implementation details. They are central to whether a deep model can actually be trained. The theoretical expressivity of the block is useless if gradients do not flow well through depth.

6. **MLP expansion and activation**: The hidden dimension inside the MLP is often larger than \\(d\\), sometimes by a factor of four or more. This lets the model expand into a richer feature space, apply a nonlinear gating or activation, and then project back. The MLP is therefore a major source of expressive power, not just an accessory to attention.

## Takeaways

A transformer layer can be viewed as the repeated application of two maps to a token state \\(X\\):

\\[
\\text{route across positions} \\quad \\longrightarrow \\quad \\text{rewrite within positions}.
\\]

More concretely:

\\[
X \\mapsto X + \\mathrm{Attn}(\\mathrm{LN}(X)) \\mapsto X + \\mathrm{Attn}(\\mathrm{LN}(X)) + \\mathrm{MLP}(\\mathrm{LN}(\\cdot)).
\\]

At a high level, the design choices answer three questions:

\\[
\\text{What are the tokens?} \\qquad
\\text{Who can talk to whom?} \\qquad
\\text{How is received information rewritten?}
\\]


- **Tokenization defines the state space.**
- **Attention defines the interaction law.**
- **The MLP defines the local update rule.**
- **Residuals and LayerNorm make the whole iterative process trainable.**

A transformer is structured dynamical system on token representations. At each layer, the model decides which tokens matter to each other, aggregates their content through attention, and then refines each token locally through the MLP. Everything else, including QKV projections, heads, positional structure, and normalization, is a design choice governing the **state space**, the **interaction law**, or the **stability of the iterative update**.
