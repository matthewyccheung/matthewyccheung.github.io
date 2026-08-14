---
layout: post
title: "(Running) Notes on Language Modeling from Scratch"
math: true
---

# Overview

Tokenization: 
- what atoms the model operates on
- popular tokenizer: byte-pair encoding, breaks input into frequently occuring chunks. want efficiency (reduced context length, adaptive computation - more model capacity on interesting parts of input)
- potentially some tokenizer-free architectures that operate directly on bytes


Refinements:    
- Activation functions: ReLU, SwiGLU  
- Positional encodings: sinusoidal, RoPE
Normalization: LayerNorm, RMSNorm, QK norm, pre-norm versus post-norm  
- Attention: full, sparse/local attention, group-query attention (GQA), multi-head latent attention (MLA)
- Recurrence/state-space models/linear attention: Mamba, Gated DeltaNet
- MLP: dense, mixture of experts
- Shape (hidden dimension, depth, number of heads, number of experts)

Kernels:
- function that runs on GPU
- each primitive operation launches a standard kernel
- custom kernel make GPU faster
- idea: organize computation to minimize data movement
- examples: operator fusion (matmul+activation), tiling (flash attention)

Parallelism:
- moving data between GPUs is extremely expensive
- use classic collective operations (like apache spark: gather, reduce, all-reduce)
- shard memory (parameters, activations, gradients, optimizer scales) across GPUs
- question: how to split computation

Inference:
- goal: generate tokens given prompt
- needed for RL, test time compute, eval
- two phases: prefill (tokens->key value pairs->KV-Cache) and decoder (generate one token at a time)
- speed up: cheaper model (pruning, quantization, distillation), speculative decoding (use cheaper draft model to generate tokens then use full score in parallel), systems optimization (fused kernels, continuous batching)

Scaling laws:
- idea: too expensive to do hyperparameter tuning at full scale
- think about scaling recipe: FLOPS->hyperparameters
- run exps to compute loss at smaller losses then fit scaling law
- careful construction of scaling recipe
- parameterize the model to get hyperparameter transfer
- predictability as important as optimality
- optimal scaling laws (ISOFLOP curves: for multiple small FLOPs budgets, find optimal N, then fit scaling law). training loss vs parameters and parameters vs FLOPs depends on dataset
- Crude approximation C=6ND: C=FLOPs budget, N=number of parameters in model, D=number of training tokens
- Rule of thumb: D=20N depending on architecture 

Evaluation:
- internal: smoothness across scale, relative performance matters (perplexity etc.)
- external: measure of absolute quality for real world use case
- LMs are general: require a lot of different evals

Data curation:
- carefully curated
- fair use
- requires processing to turn into text for training

Data processing:
- transformation: convert to text
- filtering: keep high quality data, remove harmful content
- deduplication: save compute, remove memorization (bloom filters, minhash)
- data mixing: how much to upweight/downweight each source
- rewriting/synthetic data: use LM to augment real data, more similar to downstream tasks
- pre-training (large and diverse), mid-training (high quality, including long context), post-training (supervised fine-tuning: conversations, agentic tracing)

Alignment:
- full supervision
- improve with weak supervision
- why: easier to critique (what good looks like) than generate (labels)
- Generate responses from model -> score responses with human, verifier, LM judge -> update model to prefer better responses
- algorithms: PPO, DPO, GRPO
- challenges for RL at scale: hard to tune, at scale requires lots of new infra, constantly trading off system efficiency with on-policyness


# Tokenization

- raw text = unicode text
- LM places probability distribution over sequence of tokens
- need: 1) procedure to encode strings to tokens and 2) decodes tokens back to strings
- tokenizer: do this round trip

Tokenization examples:
- word and preceding space are part of the same token " world"
- word at beginning and in middle are represented differently "hello hello"
- numbers of tokenized into few digits

Measuring tokenization performance:
- compression ratio: bytes/token
- higher compression = better since attention is quadratic in sequence length
- increase compression ratio by increasing vocab size but sparsity
- modern tokenizers have 100k, 200k
- using 150k unicode is large vocab, but low compression ratio (quite rare symbols)

Tokenizer history:
- byte tokenizer: 1 byte per unicode character, vocab size small (256) but compression small
- word tokenizer: split into chunks (different words), compression ratio good but vocab huge (unbounded)

Byte pair encoding (BPE):
- train tokenizer on raw text to construct vocab tailored to data
- intuition: common sequences of bytes are represented by a single token, rare sequences represent many tokens
- sketch: start with each byte as a token, and successively merge most common pair of adjacent tokens

Future of tokenizers:
- models need to operate on chunks/abstractions of the sequence
- chunks should be variable

# Resource accounting

- understand how to estimate training time

Data types:
- float32: 1 sign, 8 exponent, 23 fraction bits = 32 bits = 4 bytes, memory usage for 4x8 tensor = 128 bytes
- float16: 1 sign, 5 exponent, 10 fraction bits = 16 bits = 2 bytes, unstable
- bfloat16: 1 sign, 7 exponents, 8 fraction bits = 16 bits = 2 bytes, more dynamic range, resolution worse but matters less for deep learning
- others fp8 and fp4, nvidia supports fp4 (nvfp4: [-6,-4,-3,...], trick is to use a separate scaling factor per block)

Implications for training:
- float32 takes lots of memory
- bfloat16 and float16 can be instable
- mix precision: bf16 for parameters, activations, gradients and f32 for optimizer states (use AMP library - wrap code and takes care automatically)

Calculating how many FLOPs needed for training:
- Spec sheet tells us how many FLOP/s a GPU is capable of
- For example, 8 H100s for 2 weeks: 1979TFLOP/s /2 (no sparsity) * (60 * 60 * 24 * 7) seconds
- forward pass Wx+b where W in mxn and x is nxp is 2mnp (mnp for matrix multiplication and mnp for addition): 2(#tokens)(#parameters) -- generalizes to transformers on first order taylor approximation
- model flops utilization (MFU): actual FLOP/s / promised FLOP/s ~ 0.5 for model models (higher if matmuls dominate)
- Summary: matrix multiplication dominates dominate, FLOP/s depend on hardware and datatype, MFU

Arithmetic intensity:
- How to compute a thing: memory -> accelerator -> computation -> outputs to memory
- depends on two things: accelerator speed (FLOP/s) and memory bandwidth (bytes/s)
- bf16 ReLU on 1024x1024 tensor:
    - Bytes that need to be moved: 2 bytes/float x 1024^2 for reading + 2 bytes/float x 1024^2 for writing = 4n
    - FLOP/s: n comparisons
    - communication time: bytes/(h100 bytes per second)
    - computation time: n / FLOP/s
    - if we assume we can perfectly overlap: ~max(communication time, computation time) ~ 1e-6
- What is the bottleneck? 
    - communication time > computation time = memory bound
    - computation time > communication time = computation bound
    - ReLU is memory bound
    - in general, we are memory bound - data movement is expensive
- Accelerator intensity = FLOP/s / Bytes/s = how much work can the accelerator do per byte transferred?
- Arithmetic intensity = FLOPS/Bytes = how much actual work per byte for this workload
    - arithmetic intesnsity < accelerator intensity = memory bound
    - accelerator intensity < arithmetic intensity = computation bound
- GeLU: bytes=4n but flops=20n (much more complicated and requires higher FLOPS), but is still memory bound, so there is no difference in computational speed
- dot product x@w=y (still assuming bf16=2 bytes):
    - read x, read w, write y: 2n+2n+2=4n+2
    - flops= n multiplications + n-1 additions = 2n-1
    - arithmetic intensity=(2n-1)/(4n+2)~1/2
- matrix vector product w@X=y (still assuming bf16=2 bytes) where w is nx1 and X is nxn:
    - read w, read X, write y: bytes=2n+2n^2+2n=2n^2+4n=2n(n+2)
    - flops=n(2n-1)
    - arithmetic intensity=flops/bytes=n(2n-1)/(2n(n+2))~1
    - during inference, matrix vector product is what happens, so inference is memory bound
- matrix multiplication matmul(nxn,nxn) X@Y=Z
    - read X, read Y, write Z: bytes=n^2+n^2+n^2=3n^2
    - flops=n^2(2n-1)
    - arithmetic intensity=flops/bytes=n^2(2n-1)/3n^2=(2n-1)/3~n/3
    - makes sense because we're sending n^2 and getting n^3
    - reason for large batch sizes
    - compute-bound
    - transformers involve big matrixes -- compute bound
- large memory bottlnecks affect MFU because it depends on how many FLOPS can be done

Computing gradients:
- For a 2 layer ReLU network ReLU(W2(ReLU(W1 X))) where X is BxD and W is DxD
- Forward pass: ~2BD^2
- Backward pass: we need to compute h1.grad (w.r.t. inputs) and w2.grad (w.r.t. weights)
    - h1.grad=matmul(h2.grad, w2)
    - w2.grad=matmul(h2.grad, h1)
    - FLOPS=2BD^2+2BD^2 (twice as expensive as forward pass)
- Considering all the layers
    - forward pass: 2(#data points)(#parameters)
    - backward pass: 4(#data points)(#parameters)
    - total: 6(#data points)(#parameters) = 6ND (works well for transformers with short context lengths)

Training loop:
- parameters use 2(#parameters) (bf16)
- activations use 2(#parameters) (bf16)
- Optimizer states use 4(#parameters) (fp32 for stability)
    - Adagrad uses 4 bytes/parameter (second order moments)
    - Adam uses 8 bytes/parameter (first and second order moments)
- Summary: #parameters=DDL and FLOPS=6(#data points)(#parameters)

Inference:
- batch x (#parameters= #layers x #dimensions) = 2BDL
- Activation checkpointing:
    - forward pass: keep only activations in subset of layers
    - backward pass: recompute the missing activations from last checkpoint
    - trades memory for compute

# Architectures

- learn from data (generalize)
- train efficiently on GPUs
- not blow up losses

Themes:
- Llama 2 with minor variations
- Last year, we saw trend towards architecture designs that make training more stable
- This year, we see trends of architecture variations that enable long context dependence

Pre vs post-norm
- post-norm has layer norms in the residual stream
    - what people realized has important implications as networks get deeper and stability issues (gradient attenuation) -- stability and signal propagation
    - introduces complicated effects because layer norming each time going through a transformer block (changes norm of gradients as we go backwards)
- pre-norm pushes layer norms outside of residual streams (before multi-head attention)
    - motivation was stability and convergence, post-norm+layer norm doesn't converge as well
    - at initialization, gradient sizes remain the same
    - almost all modern LMs use pre-norms (but BERT was post-norm)
    - keep residual streams clean
    - can have both pre- and post-norm outside the residual stream that sanwiches the MHA
    - grok, gemma 2, olmo 2 are only ones that do non-residual stream post-norm after MHA
- every time people run into stability issues, throw in layer norms

Layer norm vs RMS Norm:
- original transformer uses layer norm (GPT 1-3, OPT, GPT-J, BLOOM)
- RMSNorm: does not subtract mean and bias (Llama family, PaLM, Chinchilla, T5)
- layer norm is more expressive than RMSNorm, but RMSNorm is nice in practice because RMSNorm is faster (without much diversity loss)
- Modern explanation: faster (just as good), fewer operations (no mean calculation), fewer parameters (no bias term to store) -- about arithmetic intensity, can be up to 25% of runtime due to memory. Basically free optimization.
- Bias terms: most modern transformers do not have bias terms (memory intensive and can possibly lead to stability issues)

Activations
- ReLU: original transformer, T5, Gopher, Chincilla, OPT
- GeLU: GPT1/2/3, GPTJ, GPT-Neox, BLOOM
- SwiGLU/GeGLU: Llama, PaLM, T5 v1.1, Gemma, most models post 2023
- Gating has become really effective, but needs another matrix of the same dimension (3 total)
- Gated models tend to use smaller FF dimensions (2/3)

Serial vs Parallel blocks:
- original transformer is serial
- some modern ones use parallel: GPTJ, PaLM, GPT-NeoX, but has fallen out of popularity
- parallel improves training speed by 15% at large scales, since MLP and attention input matrix layers can be fused
- but representation takes a hit due to non-deep

Positional embeddings:
- varies a lot across architectures
- sine embeddings: original transformer, like fourier transform
- absolute embeddings: add a position vector to the embedding (GPT1/2/3, OPT)
- relative embeddings: add a vector to the attention computation (T5, Gopher, Chinchilla)
- rotary position embeddings (RoPE): (GPTJ, PaLM, LLaMA, most 2024+ models)
    - relative position embedding: inner product between embeddings is equal to a function that only depends on the relative difference
    - we want our embeddings to be invariant to absolute position
    - we know our inner products are invariant to arbitrary rotation
    - "we know" (rotate we by 0 and know by 1) vs "of course we know" (rotate we by 2 and know by 3) - relative positions still intact. inner products are the invariant to absolute.
    - in high dimensions, infinite number of ways to rotate vector, so just pair coordinates in 2D and repeat the rotation. the angle they rotate rotate low frequency (slow -- long range dependence) and high frequency (neighbors)
    - in reality, create a large rotation matrix with sines and cosines

Hyperparameters:
- feedforward dim = 4 model dim (hidden dimension)
- most GLU variants after accounting for less parameters (scaled down to 2/3 parameters) d_ff=8/3 d_model
- even though we compute h many attention heads, this is not really more costly: compute XQ is nxd and then reshape to nxhxd/h for multi-head. head dimensions are generally forgiving.
- aspect ratios: d_model/n_layer, sweet spot is 100. extremely deep models are harder to parallelize and have higher latency
- vocabulary sizes: monolingual models (30-50k vocab sizes), multilingual/production systems (100-250k)

Dropout and regularization:
- arguments against: there is a lot of data (way more than parameters) hard to memorize, SGD basically only does a single pass on a corpus
- lots of models do dropout and weight decay
- why weight decay? some people argue that it isn't to control overfitting but it interacts with the learning rates (on cosine schedule, not on constant learning rate). stronger weight decay do significantly better (starts slow but convering to much better minimum)

Stability tricks:
- recently lots of attention on stable training (gradient spikes etc) - want to have low L2 norm
- softmaxes: can be ill-behaved due to exponentials/division by zero. exists in output and attention (normalize)
    - use z loss trick (devlin 2014) to encourage softmax normalizer log(Z) to be close to 0: (logZ)^2~0, finding it improves training stability
    - PaLM, OLMo2, Gemma 2, Qwen 3, OLMo3, Gemma 4 all use z loss
- attention softmax stability
    - the 'QK norm': layer norm Q and K before softmax to ensure they have the same scale
    - many other techniques
    - DCLM, OLMo2, Gemma 2, Qwen 3, OLMo3, Gemma 4
    - originally from multimodal vision models
- logit soft-capping
    - not as popular, but google specific trick
    - potentially looses performance - cannot express high confidence

Attention heads:
- group query attention/multi query attention: reduce number of heads
    - during deployment, need to pay for FLOPS and memory. 
    - during training, arithmetic intensity is O((1/k+1/(bn))^-1), so we want to increase batch size or increase sequence length
    - during deployment, total arithmetic operations bnd^2, total memory accesses bn^2d+nd^2, and arithmetic intensity is O((n/d+1/b)^-1), so we either need large batches and short sequnce lengths or large model dimensions
    - key idea is to shared K and V -- allows much less items to move to and from KV cache: total memory access bnd^2+bn^2k+nd^2, arithmetic intensity O((1/d+n/(dh)+1/b)^-1)
    - but loses significant expressive power -- tradeoff between efficiency and expressive power
    - group query is the middle-ground (key-query ratio)
    - other tricks: multihead latent attention (deepseek-v2)
- sparse or sliding window attention (SWA) (GPT4/Mistral): restricting attention pattern to reduce compute cost
    - attention scores QK^T is only computed for diagonal
- interleave "full" and long range attention (current standard trick)
    - for example every 4th layer is full attention
    - long range information without PE, short range information via RoPE and SWA: LLaMA4, Gemma 3, Gemma 4, OLMo 3.

# Attention Alternatives

- cost of attention rises quadratically with context length
- basic toolkit: local/sparse attention, systems engineering (flash attention)
- Q: are there methods that are linear with context length?

Linear attention:
- linear scale in context length
- change \rho(QK^T)V: drop the \rho (assume identity) and use associative property to get Q(K^T V)
- now looks like RNN: S_t=S_{t-1}+k_t v_t^T and y_t=q_t S_t

State space models
- generalize linear attention and add per-position weights
- S_t=\gamma S_{t-1}+k_t v_t^T and y_t=q_t S_t+v_t^T D