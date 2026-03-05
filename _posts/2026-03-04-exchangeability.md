---
layout: post
title:  "Exchangeability, Symmetry, and Conformal Prediction"
math: true
--- 

Understanding the concept of **exchangeability** is central to modern uncertainty quantification methods such as **conformal prediction**.  
Although classical statistics often assumes independent and identically distributed (i.i.d.) data, conformal prediction relies on the weaker assumption of **exchangeability**, which can hold even when observations are dependent.

This blog post explains the relationship between i.i.d. data, exchangeability, symmetry, Bayesian models, and conformal prediction.

---

# 1. Independent and Identically Distributed Data

A sequence of random variables

\[
Z_1, Z_2, \dots, Z_n
\]

is said to be **independent and identically distributed (i.i.d.)** if

\[
Z_i \sim P
\]

for all \(i\), and the joint distribution factorizes:

\[
P(Z_1,\dots,Z_n) = \prod_{i=1}^n P(Z_i).
\]

This assumption is extremely common in machine learning and statistics, as it simplifies analysis and inference.

However, i.i.d. is stronger than necessary for many statistical guarantees.

---

# 2. Exchangeability

A sequence of random variables

\[
Z_1, Z_2, \dots, Z_n
\]

is **exchangeable** if its joint distribution is invariant under permutations.

Formally, for any permutation \( \pi \),

\[
(Z_1,\dots,Z_n)
\stackrel{d}{=}
(Z_{\pi(1)},\dots,Z_{\pi(n)}).
\]

Intuitively, this means that the ordering of the variables contains no information.

If we shuffled the observations, the joint distribution would remain the same.

---

## Relationship to i.i.d.

Every i.i.d. sequence is exchangeable.

However, the converse is not true: exchangeable variables may be dependent.

Exchangeability therefore captures **symmetry** rather than independence.

---

# 3. Exchangeability with Latent Variables

A classic example of exchangeability arises when observations share a hidden variable.

Consider

\[
Z_i = \theta + \epsilon_i
\]

where

\[
\epsilon_i \sim \mathcal{N}(0,1)
\]

and

\[
\theta \sim \mathcal{N}(0,1).
\]

Conditional on \( \theta \), the observations are i.i.d.:

\[
Z_i \mid \theta \sim \mathcal{N}(\theta,1).
\]

However, marginally the variables are correlated:

\[
\mathrm{Cov}(Z_i,Z_j) = \mathrm{Var}(\theta).
\]

Despite this correlation, the sequence remains exchangeable because every variable is generated using the same mechanism.

---

# 4. de Finetti's Theorem

The connection between exchangeability and latent variables is formalized by **de Finetti's theorem**.

For an infinite exchangeable sequence \(Z_1,Z_2,\dots\), there exists a latent variable \( \Theta \) such that

\[
P(Z_1,\dots,Z_n)
=
\int
\prod_{i=1}^n P(Z_i \mid \theta)
\, dP(\theta).
\]

Thus exchangeable sequences can be represented as **mixtures of i.i.d. sequences**.

In other words, the observations are i.i.d. once we condition on a hidden variable.

This idea plays a central role in **Bayesian statistics**, where parameters such as \( \theta \) are treated as random variables.

---

# 5. Exchangeability in Bayesian Regression

Consider Bayesian linear regression:

\[
y_i = x_i^\top \beta + \epsilon_i
\]

with

\[
\epsilon_i \sim \mathcal{N}(0,\sigma^2)
\]

and prior

\[
\beta \sim p(\beta).
\]

Conditional on \( \beta \), the data are independent:

\[
y_i \mid \beta \sim \mathcal{N}(x_i^\top\beta,\sigma^2).
\]

However, after integrating over \( \beta \), the observations become correlated:

\[
\mathrm{Cov}(y_i,y_j) = \mathrm{Var}(\beta^\top x_i).
\]

Nevertheless, the observations remain exchangeable because the generative process treats them symmetrically.

---

# 6. Why Exchangeability Matters for Conformal Prediction

Conformal prediction relies on a key property of exchangeable sequences.

Let

\[
S_1,\dots,S_n,S_{n+1}
\]

be **exchangeable conformity scores**.

Then the rank of the new score

\[
S_{n+1}
\]

among the \(n+1\) scores is uniformly distributed:

\[
\mathrm{rank}(S_{n+1}) \sim \text{Uniform}\{1,\dots,n+1\}.
\]

This implies

\[
P(S_{n+1} \le q_{1-\alpha}) \ge 1-\alpha,
\]

where \(q_{1-\alpha}\) is the empirical quantile of the calibration scores.

This simple rank argument is the foundation of conformal prediction.

---

# 7. Model Misspecification

A remarkable property of conformal prediction is that it remains valid even when the predictive model is incorrect.

Suppose the true data generating process is

\[
y = f(x) + \epsilon,
\]

but we use a misspecified model \( \hat{y}(x) \).

Define conformity scores

\[
S_i = |y_i - \hat{y}(x_i)|.
\]

As long as the scores are exchangeable, the conformal coverage guarantee still holds:

\[
P(Y_{n+1} \in C(X_{n+1})) \ge 1-\alpha.
\]

The intervals may become wider when the model is poor, but coverage remains correct.

---

# 8. Key Takeaways

- **i.i.d. data are exchangeable**, but exchangeable data need not be independent.
- Exchangeability captures **symmetry of the data-generating process**.
- By de Finetti's theorem, exchangeable sequences are **mixtures of i.i.d. sequences**.
- Many Bayesian models naturally produce exchangeable observations.
- Conformal prediction relies only on exchangeability, not on model correctness.

Because of this minimal assumption, conformal prediction provides **distribution-free uncertainty guarantees** that remain valid even when the predictive model is misspecified.

---

Exchangeability therefore provides the conceptual bridge connecting **Bayesian statistics, symmetry assumptions, and modern uncertainty quantification methods like conformal prediction**.