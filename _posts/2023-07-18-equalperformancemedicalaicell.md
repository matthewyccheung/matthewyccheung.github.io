---
layout: single
classes: wide
author_profile: true
title: "[Paper Insights] The path toward equal performance in medical machine learning"
show_date: true
comments: true
---

The path toward equal performance in medical machine learning

[Paper link, DOI: 10.1016/j.patter.2023.100790](https://www.cell.com/patterns/pdf/S2666-3899(23)00145-9.pdf)

Summary: The paper analyzes why medical AI algorithms can be biased (i.e. imbalanced datasets). The paper does an analysis through the bias-variance equation conditioned on group, and finds that we can only get so fair algorithmically and that we need to get better data. My interpretation of this article from an ECE perspective: we will always be limited by the data quality. We should make data quality better through better hardware/data collection processes.

<h2> Motivation from interesting results </h2>

1. Training model only on women: model performance on women still worse than on men
2. Increasing the representation of women (while keeping the dataset size fixed): slightly improved model performance for both men and women. We’d think that model performance would decline in men. May be due to “better” data in men than in women.

<div style="text-align: center"><img src="/images/example_equalperformancemedicalaicell.png" style="width: 70%; height: 70%"/></div>

<h2> Does underrepresentation always cause a disproportionate reduction in model performance? Not necessarily </h2>

1. If inputs are sufficiently informative and the model is expressive enough to represent mapping for all groups, training dataset minority does not necessarily lead to suboptimal performance.
2. If the model is highly flexible and groups differ significantly, majority samples cannot provide helpful regularization for minority classes. Either the model or data are not sufficiently expressive to capture optimal decision boundaries. Empirical risk minimization will always be biased towards the majority.

<h2> Optimal for each group ≠ equal performance for each group </h2>
- This is caused by the intrinsic difficulty of estimation tasks for the different groups either from:
1. differing input disturbance characteristics w.r.t. the underlying physiological property of interest and disparities in the equipment/medical environment 
2. unobserved confounders (i.e. hormonal level changes)

<h2> Existing methods </h2>
- Argues that most existing methods like domain adaptation have limited empirical success in mitigating effects of underrepresentation in the medical domain and often result in lower overall performance
- Because these methods cannot address differences in task difficulty.

<h2> Recommendations </h2>
1. Work in close collaboration with medical experts, as it will typically require identifying or developing appropriate additional (or alternative) measurement modalities
2. Need improved root cause diagnosis methods - quantifying task difficulty, determining if the issue is model expressivity, etc.

<div style="text-align: center"><img src="/images/summary_equalperformancemedicalaicell.png" style="width: 70%; height: 70%"/></div>
<br>
<div style="text-align: center"><img src="/images/summary_equalperformancemedicalaicell2.png" style="width: 70%; height: 70%"/></div>