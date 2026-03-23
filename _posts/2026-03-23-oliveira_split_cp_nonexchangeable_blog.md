---
layout: post
title:  "Split Conformal Prediction and Non-Exchangeable Data"
math: true
--- 

*Based on Oliveira, Orenstein, Ramos, and Romano, [**“Split Conformal Prediction and Non-Exchangeable Data”** (JMLR, 2024).*](https://jmlr.org/papers/v25/23-1553.html)

Conformal prediction is famous for one simple promise: if your data are exchangeable, then your prediction sets achieve the target coverage level without strong modeling assumptions. But many real datasets are not exchangeable at all. Time series are temporally dependent. Spatial data are correlated across nearby locations. Spatiotemporal data combine both. The main message of Oliveira et al. is that **standard split conformal prediction often still works surprisingly well under dependence**, and there is a clean way to quantify the price paid for non-exchangeability.

The paper replaces exact exchangeability arguments with two kinds of corrections:

- a **concentration correction** that measures how well the calibration empirical CDF tracks its population counterpart;
- a **decoupling correction** that measures how far a test point is from behaving like an independent fresh draw.

These are summarized by small error terms such as $\epsilon_{\mathrm{cal}}$, $\epsilon_{\mathrm{train}}$, and $\epsilon_{\mathrm{test}}$, together with failure probabilities like $\delta_{\mathrm{cal}}$ and $\delta_{\mathrm{test}}$.

## Standard Split CP Setup

Suppose we observe data

$$
(X_i, Y_i)_{i=1}^n,
$$

split into a training set $I_{\mathrm{train}}$, a calibration set $I_{\mathrm{cal}}$, and a test set $I_{\mathrm{test}}$. We fit a predictor on the training set and define a nonconformity score

$$
\widehat s_{\mathrm{train}}(x,y) := s\big((X_i,Y_i)_{i\in I_{\mathrm{train}}}, (x,y)\big).
$$

For example, in regression one can use an absolute residual score,

$$
\widehat s_{\mathrm{train}}(x,y) = |y - \widehat\mu(x)|.
$$

Then the empirical calibration quantile is

$$
\widehat q_{\phi,\mathrm{cal}} := \inf\left\{t\in\mathbb R:
\frac{1}{n_{\mathrm{cal}}}\sum_{i\in I_{\mathrm{cal}}}
\mathbf 1\{\widehat s_{\mathrm{train}}(X_i,Y_i)\le t\} \ge \phi\right\}.
$$

The split conformal prediction set at level $\phi$ is

$$
C_\phi(x) := \{y\in\mathcal Y: \widehat s_{\mathrm{train}}(x,y) \le \widehat q_{\phi,\mathrm{cal}}\}.
$$

Under exchangeability, choosing $\phi = 1-\alpha$ yields coverage near $1-\alpha$. The paper asks what survives when exchangeability fails.

## Non-exchangeability Penalty

The paper defines, for any training-dependent threshold $q_{\mathrm{train}}$,

$$
P_{q,\mathrm{train}} := \mathbb P\big[\widehat s_{\mathrm{train}}(X_*,Y_*) \le q_{\mathrm{train}} \,\big|\, (X_i,Y_i)_{i\in I_{\mathrm{train}}}\big],
$$

where $(X_\*,Y_\*)$ is an idealized fresh draw independent of the training set.

The key assumptions are:

1. The empirical calibration CDF at the training-dependent threshold is close to the population probability, meaning the calibration sample should still estimate the score distribution reasonably well.

$$
\mathbb P\left[
\left|
\frac{1}{n_{\mathrm{cal}}}\sum_{i \in I_{\mathrm{cal}}}
\mathbf 1\{\widehat s_{\mathrm{train}}(X_i,Y_i) \le q_{\mathrm{train}}\}
- P_{q,\mathrm{train}}
\right| \le \epsilon_{\mathrm{cal}}
\right] \ge 1-\delta_{\mathrm{cal}}.
$$

2. Test decoupling or test concentration: For marginal guarantees, the test point should behave approximately like an independent fresh draw:

$$
\left|
\mathbb P[\widehat s_{\mathrm{train}}(X_i,Y_i)\le q_{\mathrm{train}}]
-
\mathbb P[\widehat s_{\mathrm{train}}(X_*,Y_*)\le q_{\mathrm{train}}]
\right|
\le \epsilon_{\mathrm{train}}.
$$

This is the paper’s replacement for exact exchangeability on the test side: even though data are dependent, the distribution of the test residual is close to that of a fresh point.

3. For empirical guarantees over the whole test block, one instead assumes concentration of the **test empirical CDF** around the same population target, with error $\epsilon_{\mathrm{test}}$ and failure probability $\delta_{\mathrm{test}}$.
This gives a very useful mental model:
- $\epsilon$ terms measure **how wrong the exchangeability heuristic is**;
- $\delta$ terms measure **how often the concentration event fails**.

## Marginal versus empirical guarantees

This distinction is one of the most important points in the paper.

**Marginal coverage** concerns a single test point:
$$
\mathbb P\big[Y_i \in C_{1-\alpha}(X_i)\big].
$$ This is an average-probability statement over the randomness in the data-generating process. It says that if you sample one test example from the test regime, the prediction set covers with high probability.

**Empirical coverage** concerns the **fraction of covered points in the realized test set**:
$$
\frac{1}{n_{\mathrm{test}}}\sum_{i\in I_{\mathrm{test}}}
\mathbf 1\{Y_i \in C_{1-\alpha}(X_i)\}.
$$
This is stronger in a different sense: it controls the actual realized coverage rate over a block of test points, not just the average for one point. In dependent data, these are not the same because many positively correlated successes or failures can happen together.

The paper unifies them by showing how both degrade through explicit $\epsilon$ and $\delta$ corrections**.

## Theorem 1: marginal coverage under non-exchangeability

Theorem 1 states that if the calibration concentration condition and the test decoupling condition hold, then for every test index $i$,

$$
\mathbb P\big[Y_i \in C_{1-\alpha}(X_i)\big]
\ge 1-\alpha-\epsilon_{\mathrm{cal}}-\delta_{\mathrm{cal}}-\epsilon_{\mathrm{train}}.
$$

If the score distribution is conditionally continuous, then the deviation from the nominal level is bounded on both sides:

$$
\left|
\mathbb P\big[Y_i \in C_{1-\alpha}(X_i)\big] - (1-\alpha)
\right|
\le
\epsilon_{\mathrm{cal}}+\delta_{\mathrm{cal}}+\epsilon_{\mathrm{train}}.
$$

Interpretation: the classical split conformal proof is no longer exact, but it is approximately correct up to three penalties:

1. calibration estimation error $\epsilon_{\mathrm{cal}}$;
2. calibration failure probability $\delta_{\mathrm{cal}}$;
3. mismatch between the test point and an independent fresh point, $\epsilon_{\mathrm{train}}$.

## Theorem 2: empirical coverage over the test block

Theorem 2 replaces the single-point decoupling assumption with concentration of the test empirical CDF. It shows that, with probability at least $1-\delta_{\mathrm{cal}}-\delta_{\mathrm{test}}$,

$$
\frac{1}{n_{\mathrm{test}}}\sum_{i\in I_{\mathrm{test}}}
\mathbf 1\{Y_i \in C_{1-\alpha}(X_i)\}
\ge 1-\alpha-\eta,
\qquad
\eta = \epsilon_{\mathrm{cal}} + \epsilon_{\mathrm{test}}.
$$

Under conditional continuity of the score distribution, the realized empirical coverage stays within $\eta$ of $1-\alpha$ with probability at least $1-2\delta_{\mathrm{cal}}-2\delta_{\mathrm{test}}$.

The important idea is that **empirical coverage does not depend on a pointwise test-decoupling term**. Instead, it depends on whether the whole test block has a stable empirical score distribution.

## Theorem 3: conditional coverage on sets

The paper also extends split conformal guarantees to set-conditional statements. For a family of subsets $\mathcal A \subseteq \mathcal X$, it defines localized calibration quantiles and proves that for each $A\in\mathcal A$,

$$
\mathbb P\big[Y_i \in C_{1-\alpha}(X_i;A) \mid X_i\in A\big]
\ge 1-\alpha-\epsilon_{\mathrm{cal}}-\delta_{\mathrm{cal}}-\epsilon_{\mathrm{train}}.
$$

Again, with continuity, the deviation from $1-\alpha$ is bounded in absolute value by the same correction terms.

This is not full conditional coverage in the impossible sense of conditioning on $X=x$, but rather a practical group-conditional guarantee over a structured collection of sets.

## $\epsilon$ and $\delta$ under $\beta$-mixing

A major strength of the paper is that it does not stop at abstract assumptions. It verifies them for **stationary $\beta$-mixing processes**, a broad class including many Markov, hidden Markov, and ARMA-type systems. The paper derives $\epsilon$ and $\delta$ equations for different applications.

Theorems 4 and 5 instantiate the general framework for stationary $\beta$-mixing data. In this setting:

- the marginal coverage bound becomes
  $$
  \mathbb P[Y_i\in C_{1-\alpha}(X_i)] \ge 1-\alpha-\eta,
  $$
  with $\eta = \epsilon_{\mathrm{cal}}+\epsilon_{\mathrm{train}}+\delta_{\mathrm{cal}}$;
- the empirical coverage bound becomes
  $$
  \frac{1}{n_{\mathrm{test}}}\sum_{i\in I_{\mathrm{test}}}\mathbf 1\{Y_i\in C_{1-\alpha}(X_i)\}
  \gtrsim 1-\alpha-(\epsilon_{\mathrm{cal}}+\epsilon_{\mathrm{test}})
  $$
  with high probability.

When the mixing coefficients decay fast enough, these error terms can have the **same asymptotic order as the iid bounds**. That is the paper’s strongest theoretical message: some dependent processes are not much worse than iid for split conformal purposes.

## Takeaways

Exchangeability is not the only route to conformal validity. What split conformal really needs is that calibration scores and test scores behave enough like samples from the same score distribution, and that this behavior can be quantified by concentration plus decoupling.

That viewpoint unifies several regimes:
- iid data: all correction terms are the familiar concentration errors
- weakly dependent data: add small dependence penalties
- strongly dependent data: those penalties can become large, and coverage may deteriorate

So the paper does not claim that conformal prediction is magically distribution-free under arbitrary dependence. Rather, it shows that **non-exchangeability can be turned into explicit, auditable error terms**.

For people already using split CP, the practical lesson is reassuring:
- you often do **not** need a radically new conformal algorithm just because your data are dependent
- standard split CP can remain competitive when dependence is moderate
- the main question is whether your calibration and test blocks are “effectively large enough” after accounting for dependence

In other words, dependence reduces the **effective sample size**, and the $\epsilon$ corrections are a mathematical way to account for that loss.