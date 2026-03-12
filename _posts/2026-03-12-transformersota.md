---
layout: post
title:  "Transformers 1-100: From Seminal Papers to Modern Standard Practice"
math: true
--- 

A transformer is a token mixer whose modern design choices are mostly about stabilizing depth, encoding geometry, and making routing cheap at scale.
The seminal works established the basic language of token-based communication.
Modern design choices refine that language so it trains more stably, uses context more effectively, and serves faster on real hardware.
This post walks through the design of transformers to understand which design choices are *structural*, which are *historical*, and which are now *default* in modern large-scale models.

Throughout, let the input token matrix be

\\[
X \\in \\mathbb{R}^{n \\times d},
\\]

where \\(n\\) is the number of tokens and \\(d\\) is the model width.
A transformer is best understood as an **iterated routing-and-rewriting machine**:

\\[
\\text{route information across positions} \\;\\Longrightarrow\\; \\text{rewrite features within each position}.
\\]

More concretely, each block alternates between a self-attention sublayer and an MLP sublayer. Self-attention decides **who talks to whom**; the MLP decides **how each token updates its local feature state after receiving context**.

---

## Preliminaries

The cleanest expert mental model is that a transformer block has three ingredients:

\\[
\\text{state space} \\; + \\; \\text{interaction law} \\; + \\; \\text{update rule}.
\\]

- The **state space** is the tokenization and embedding choice: what counts as one atomic object.
- The **interaction law** is attention: how tokens exchange information.
- The **update rule** is the residual-plus-MLP structure: how contextualized features are accumulated and transformed stably over depth.

If we define

\\[
Q = X W_Q, \\qquad K = X W_K, \\qquad V = X W_V,
\\]

then single-head scaled dot-product attention is

\\[
\\operatorname{Attn}(X) = \\operatorname{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}} + B\\right)V,
\\]

where \\(B\\) is any positional or structural bias term.
This equation is the heart of the transformer: each token becomes a **data-dependent weighted average of other tokens' value vectors**.

A modern pre-normalized block is usually written as

\\[
X_{\\text{attn}} = X + \\operatorname{Attn}(\\operatorname{Norm}(X)),
\\]

\\[
X' = X_{\\text{attn}} + \\operatorname{MLP}(\\operatorname{Norm}(X_{\\text{attn}})).
\\]

So the block alternates between:

1. **cross-token routing** via attention, and
2. **within-token feature transformation** via the MLP.

That is the design pattern that has survived essentially every major generation of transformers.

The Query-Key-Value construction is not arbitrary. It is a decomposition of contextual lookup into three roles:

- **Query \\((Q)\\):** what the current token is looking for.
- **Key \\((K)\\):** what each other token advertises about itself.
- **Value \\((V)\\):** the actual content passed along if that token is selected.

Mathematically, for token \\(i\\) attending to token \\(j\\), the compatibility score is

\\[
s_{ij} = \\frac{q_i^{\\top} k_j}{\\sqrt{d_k}},
\\]

and the normalized attention weight is

\\[
a_{ij} = \\frac{\\exp(s_{ij})}{\\sum_{t=1}^{n} \\exp(s_{it})}.
\\]

The contextualized representation of token \\(i\\) is then

\\[
z_i = \\sum_{j=1}^{n} a_{ij} v_j.
\\]

**Queries choose, keys match, values deliver**. A transformer layer is therefore a learned retrieval system whose retrieval rule is recomputed on the fly from the current hidden state.

---

## Attention is all you need

The original paper, *Attention Is All You Need*, replaced recurrence and convolutions with attention in an encoder-decoder architecture ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).
Its most important contribution was not merely a new layer, but a new computational philosophy:

\\[
\\text{sequence modeling} \\approx \\text{parallel content-based communication over tokens}.
\\]

Historically, the seminal design choices in that paper were:

- multi-head self-attention,
- residual connections,
- layer normalization,
- position-wise feed-forward sublayers,
- positional encodings,
- and encoder-decoder cross-attention.

Multi-head attention can be written as

\\[
\\operatorname{MHA}(X) = \\operatorname{Concat}(H_1, \\dots, H_h) W_O,
\\]

where each head is

\\[
H_r = \\operatorname{softmax}\\!\\left(\\frac{Q_r K_r^{\\top}}{\\sqrt{d_k}} + B_r\\right)V_r.
\\]

The expert interpretation is that multiple heads let the model learn **multiple interaction subspaces** in parallel: syntactic, semantic, local, global, structural, and so on.

---

## Extensions to encoder-only, decoder-only, and encoder-decoder

After the original transformer, three major architectural branches became standard.

1. **Encoder-only (BERT, [Devlin et al., 2018](https://aclanthology.org/N19-1423/))**: BERT made the encoder-only transformer the standard architecture for bidirectional representation learning. The key idea was masked language modeling (MLM): hide some tokens and predict them from both left and right context. [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) did not use MLM.  Instead, it was trained as a sequence-to-sequence model using a causal (autoregressive) language modeling objective, where the model predicts the next token in the output sequence based on previous tokens—commonly used in tasks like machine translation. The encoder-only transformer is best for representation learning and understanding tasks.

2. **Decoder-only (GPT, [Radford et al., 2018](https://www.mikecaptain.com/resources/pdf/GPT-1.pdf))**: GPT-style models use causal masked self-attention, so token \\(i\\) can only attend to tokens \\(j \\le i\\). In other words, each token can only attend to previous tokens in the sequence, ensuring the model generates text in a left-to-right, sequential manner. This prevents information leakage from future tokens. This is implemented with a causal mask:
\\[
B_{ij} =
\\begin{cases}
0, & j \\le i, \\\\\\\\
-\\infty, & j > i
\\end{cases}
\\]
GPT-3 ([Brown et al., 2020(https://arxiv.org/abs/2005.14165)) demonstrated that very large language models trained on vast amounts of unlabeled text can perform a wide range of tasks with minimal task-specific training, effectively acting as meta-learners. In other words, the scale-and-prompting regime showed that large autoregressive transformers could perform strong few-shot learning from context alone. GPT-style architectures/decoder-only transformers are best for autoregressive generation and in-context learning.

3. **Encoder-decoder (T5, [Raffel et al., 2019](https://arxiv.org/abs/1910.10683)**: T5 made the encoder-decoder transformer into a unified text-to-text transfer learning framework. The expert view is that encoder-decoder models are especially natural when the input and output play different roles, such as translation, summarization, or structured conditional generation.

The first major design choice is:
- Understanding and retrieval \\(\\to\\) encoder-only.
- Free-form next-token generation \\(\\to\\) decoder-only.
- Conditional transduction \\(\\to\\) encoder-decoder.

---

## Scale

A major shift in expert thinking came from the scaling era: GPT-3 ([Brown et al., 2020](https://arxiv.org/abs/2005.14165)), PaLM ([Narang et al., 2022](https://arxiv.org/abs/2204.02311)), and related models showed that architecture alone was not the story. The deeper lesson was:

\\[
\\text{performance} = f(\\text{architecture}, \\text{data}, \\text{objective}, \\text{compute}, \\text{optimization}).
\\]

Researchers stopped asking only, “What layer should I use?” and started asking, “What architecture continues to improve smoothly under more data and more compute?”. Modern design choices are best understood not as isolated tricks, but as **choices that preserve training stability and inference efficiency at scale**.

---

## Vision Transformers

Vision Transformers (ViT, [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)) showed that transformers could work extremely well in vision by treating image patches as tokens. If an image is divided into \\(n\\) patches, the same token matrix view applies:

\\[
X \\in \\mathbb{R}^{n \\times d}.
\\]

The important conceptual change was this:

\\[
\\text{the transformer is not inherently linguistic; it is a general token processor.}
\\]

Tokenization choice is foundational. In text, tokens are subwords or bytes; in images, they may be patches; in multimodal systems, they may include both text tokens and visual tokens; in scientific data, they may be spatial cells, points, or latent patches.

---

## Modern Design Choices

The original transformer gave the blueprint, but the dominant modern stack has shifted in several important ways.
The most common defaults in modern decoder-only large language models are:

\\[
\\text{Pre-Norm or RMSNorm} + \\text{RoPE} + \\text{SwiGLU} + \\text{GQA/MQA} + \\text{FlashAttention}.
\\]

Each of these solves a specific systems or optimization problem.

#### Normalization

1. **Post-LN** ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)): The original transformer used what is now called **Post-LN**, where normalization is applied after the residual addition. 
2. **Pre-LN** ([Xiong et al., 2020](https://arxiv.org/abs/2002.04745)): Later work showed that moving normalization inside the residual branch, yielding , greatly improves optimization stability, especially in deep transformers.
A Pre-LN block takes the form
\\[
X_{\\text{attn}} = X + \\operatorname{Attn}(\\operatorname{LN}(X)),
\\]
\\[
X' = X_{\\text{attn}} + \\operatorname{MLP}(\\operatorname{LN}(X_{\\text{attn}})).
\\]
Layer normalization itself is
\\[
\\operatorname{LN}(x) = \\gamma \\odot \\frac{x - \\mu(x)}{\\sigma(x)} + \\beta,
\\]
where \\(\\mu(x)\\) and \\(\\sigma(x)\\) are computed across the feature dimension of a token, and \\(\\gamma\\), \\(\\beta\\) are learned parameters.
3. **RMSNorm** ([Zhang and Sennrich, 2019](https://arxiv.org/abs/1910.07467)): Modern large decoder-only models often go one step further and replace LayerNorm with RMSNorm, which drops the mean-centering step and normalizes only by the root mean square .
\\[
\\operatorname{RMSNorm}(x) = \\gamma \\odot \\frac{x}{\\sqrt{\\frac{1}{d}\\sum_{m=1}^{d} x_m^2 + \\varepsilon}}.
\\]

Generally, Pre-LN or RMSNorm are used when depth and scale make optimization the bottleneck.
Pre-LN generally ensures stable training via early normalization. 
RMSNorm generally offers a faster, equivalent alternative when used in Pre-LN architectures. 
The two are arithmetic equivalents in practice, enabling seamless substitution for efficiency gains ([Jiang et al,. 2023](https://arxiv.org/abs/2305.14858)).

#### Positional structure

Attention alone is permutation-equivariant, so the model needs positional structure. The original transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) added sinusoidal positional encodings to token embeddings.

1. **Sinusoidal Encodings**: Each position in a sequence is encoded using a combination of sine and cosine functions with different frequencies, where each dimension of the embedding corresponds to a sinusoid with a wavelength forming a geometric progression (from $2\pi$ to $10000 \cdot 2\pi$). The encoding is calculated as: 
\\[PE(pos,2i)=sin\\Bigl(\\frac{pos}{10000^{2i/d}}\\Bigr), 
\\qquad
PE(pos,2i+1)=cos\\Bigl(\\frac{pos}{10000^{2i/d}}\\Bigr)\\]
2. **Rotary Position Embedding (RoPE)** ([Su et al., 2021](https://arxiv.org/abs/2104.09864))
A major practical evolution was RoPE, which rotates queries and keys by position-dependent matrices. Abstractly, if \\(R_m\\) is the rotation associated with position \\(m\\), then RoPE replaces
\\[
q_m \\mapsto R_m q_m, \\qquad k_m \\mapsto R_m k_m.
\\]
This makes the attention score depend naturally on relative position through the interaction of the rotated vectors. RoPE injects position into the geometry of matching itself, not just into the input features.
That is one reason it became a common default in modern decoder-only LLMs like LLaMA ([Touvron et al., 2023](https://arxiv.org/abs/2302.13971)), OPT ([Zhang et al., 2022](https://arxiv.org/abs/2205.01068)), and models like BLOOM ([Scao et al., 2022](https://arxiv.org/abs/2211.05100)) and PaLM ([Narang et al., 2022](https://arxiv.org/abs/2204.02311)).

Sinusoidal PEs mix content and location, relying on the model to subtract positions to get relative distances.
But RoPE encodes position in the angle of rotation, not as a fixed location like sinudoidal PEs. When two tokens interact (dot product), their relative position emerges naturally from the angle difference.

#### MLP

MLP is an important design lever. A classical position-wise MLP is

\\[
\\operatorname{MLP}(x) = W_2 \\phi(W_1 x + b_1) + b_2,
\\]

where \\(\\phi\\) might be ReLU or GELU.

A major improvement was the use of gated linear unit variants.
Gated MLPs work better than standard MLPs because they introduce a spatial gating mechanism that enables cross-token communication—a capability essential for tasks like language and vision, where relationships between elements (words, image patches) matter. 
1. The input sequence is split into two parts along the channel dimension: $X_1$ (the data to be updated) and $X_2$ (the gating signal)
2. A linear projection (often initialized near zero) is applied to $W_2$, producing a gating signal $f(WX_2+b)$
3. This signal is element-wise multiplied with $X_1$, resulting in $X_{out} = X_1 \\odot f(WX_2+b)$. 
The multiplication acts like a learned mask: only the parts of $X_1$ that are deemed relevant by the gating signal are preserved, while irrelevant parts are suppressed. 
Because the linear projection operates across the sequence dimension (i.e., across tokens), the gating signal integrates information from all positions, enabling global context without computing pairwise attention scores. 

The most used are GEGLU and SwiGLU ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)).
A generic gated form is

\\[
\\operatorname{GLU\\text{-}MLP}(x) = \\big(\\phi(xW) \\odot xV\\big) W_2.
\\]

SwiGLU replaces \\(\\phi\\) with the Swish / SiLU nonlinearity.

The key idea is: attention routes information; gated MLPs control feature-selective amplification after routing. 
This is why modern large models often use SwiGLU rather than a plain ReLU MLP.

## Attention heads at inference time

The original transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) used full multi-head attention (MHA), with separate query, key, and value projections per head.
At inference time, especially for autoregressive decoding, the key-value cache becomes a major memory-bandwidth bottleneck.

1. **Multi-Query Attention** ([MQA, Shazeer, 2020](https://arxiv.org/abs/2002.05202)): shares one set of keys and values across all query heads 
2. **Grouped-Query Attention** ([GQA, Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)): uses an intermediate number of key-value heads, trading off quality and speed

At a high level, if there are \\(h\\) query heads but only \\(g\\) key-value heads with \\(g < h\\), then multiple query heads share the same key-value group. MHA maximizes expressivity, MQA maximizes decoding efficiency, GQA is the practical compromise. This compromise is now common in modern open LLMs, including larger Llama 2 variants ([Touvron et al., 2023](https://arxiv.org/abs/2302.13971)).

## Flash Attention

A major systems lesson of the last few years is that the formula for attention is not the whole story.
Standard attention has quadratic memory traffic in sequence length, and memory movement often dominates wall-clock cost.
This is because we need to constantly run to the store: load data, compute a step, write result back and repeat.
Flash Attention ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) is an optimized algorithm that dramatically speeds up transformer models by rethinking how attention computations are performed on GPUs, especially addressing the memory bottleneck caused by frequent data movement between slow high-bandwidth memory (HBM) and fast on-chip SRAM.
Flash Attention reorders the computation with tiling so that exact attention is computed with much better IO behavior [Dao et al., 2022].
The key idea is that asymptotic formula for attention is not enough; hardware-aware implementation changes what is practical.
Today, efficient exact attention kernels such as FlashAttention are part of the standard engineering stack for large transformers.
Flash Attention keeps data local, minimizes memory transfers, and fully uses the GPU’s compute power. 
This results in faster training and inference, lower memory usage, and support for much longer sequences, while producing exact same results as vanilla attention. 
The key ingredientwsare the following:

1. Tiling: bring a small batch of "tiles" onto the counter at a time
2. Online Softmax: compute the final result incrementally — no need to store the full attention matrix. 
3. Recomputation: Instead of saving intermediate results, you re-compute them when needed — the cost of recomputing is cheaper than fetching it from memory.

## Mixture-of-Experts

One of the most important research-and-production design choices beyond dense transformers is **Mixture-of-Experts** ([MoE, Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)) replaces the standard dense feed-forward network (FFN) in each Transformer layer with a collection of smaller, specialized subnetworks called experts, along with a router that dynamically assigns each input token to one or more of these experts.
Switch Transformer showed that sparse routing can increase parameter count dramatically without a proportional increase in per-token compute ([Fedus et al., 2021](https://arxiv.org/abs/2101.03961)).
This sparse activation allows models to have massive total parameter counts (e.g., 141B in Mixtral-8x22B) while only using a fraction (e.g., ~39B) per token — enabling high capacity with low compute.
The router (a lightweight network) analyzes each token and assigns it to the top-k experts (e.g., top-2) using a softmax-based scoring system. 
Only the selected experts compute outputs, which are then combined via weighted sums based on the router’s scores.
MoE buys capacity without paying dense compute everywhere. 

Briefly, if \\(E_1, \\dots, E_M\\) are expert functions and \\(r(x)\\) is a router over experts, then an MoE layer has the abstract form

\\[
\\operatorname{MoE}(x) = \\sum_{m=1}^{M} r_m(x) E_m(x),
\\]

with only a few \\(r_m(x)\\) active in practice.

MoE remains a major route to pushing parameter scale; in industry, it is used when serving and routing complexity are justified by the gain.

---

## Standard models today

For a modern large **decoder-only** transformer, the most common expert defaults are roughly:

\\[
\\text{causal decoder-only stack} + \\text{Pre-Norm/RMSNorm} + \\text{RoPE} + \\text{SwiGLU} + \\text{GQA} + \\text{KV cache} + \\text{FlashAttention}.
\\]

For a modern **encoder-only** transformer, the stack is more likely to preserve full bidirectional attention and may keep more classical choices depending on task and scale.
For a modern **vision transformer**, the defaults usually shift to patch tokenization, 2D positional structure, and application-specific head designs.

So the right expert takeaway is not that there is one universal transformer, but that there is now a **family of stable defaults** tuned to regime:

\\[
\\text{regime} = \\text{task} + \\text{scale} + \\text{training objective} + \\text{inference constraints}.
\\]

---

## Takeaways

#### Design
1. Tokenization: match the natural atomic units of the data. Subwords, bytes, image patches, latent patches, multimodal token streams, or domain-specific units all change the representation geometry.
2. Information flow: Bidirectional versus Causal versus Conditional encoder-decoder
3. Positional bias: absolute, relative, rotary, multidimensional - how geometry is injected into attention.
4. Normalization: Pre-LN and RMSNorm
5. Efficiency improvements: GQA, KV caching, FlashAttention, and MoE

#### Lessons from prior works
1. **Seminal papers** taught us that attention can replace recurrence, that different masking patterns induce different model families, and that scale changes what these models can do.
2. **Modern practice** teaches us that normalization placement, positional structure, MLP gating, key-value sharing, and hardware-aware attention kernels are what make transformers work well at large scale.
3. **Research frontier work** extends this further with sparse routing, longer context mechanisms, multimodal tokenization, and increasingly specialized architectural priors.