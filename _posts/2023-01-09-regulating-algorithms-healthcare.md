---
layout: single
classes: wide
author_profile: true
title: "Regulating machine learning for healthcare is not much different than regulating traditional medical devices"
show_date: true
comments: true
---

Recently I came across an [article](https://ai.jmir.org/2023/1/e42940/PDF) on whether the current FDA regulation for ML in healthcare sufficient. The authors argue that:

> The current benchmark for approval requires companies to demonstrate good model performance on a varied data set and in a real-world setting. With no explicit definition of what constitutes reproducible standards, it is no surprise that the current FDA-approved ML algorithms vary considerably by the size of data sets and number of sites

> Under the current regulation, once an algorithm is approved, its behavior will remain fixed, defeating the distinguishing advantage of many adaptive algorithms in their ability to learn throughout their life cycle. Its current inflexible state not only reduces its clinical utility but can alarmingly infringe patient safety.

> Additionally, the current FDA framework does not regulate the inception of an ML algorithm. As a result, a number of algorithms have been approved, many of similar use cases with varying development sites and data sizes. This can potentially constitute an inefficient use of resources

The authors propose an framework for the evolution of clinical algorithms that mirror training of medical clinicians. Interestingly, as Ashu (Rice ECE faculty) pointed out, this is not unique to ML algorithms, In fact, any medical device that contains a model, regardless of how complex it is, has this issue. The model can be a simple linear regression. All those algorithms are fixed and based on some clinical trial data and could present safety issues. Moreover, a rudimentary model can change over time based on new data. If the FDA has been regulating medical devices and algorithms for years, how is it different to regulating the new generation of ML algorithms? I'm not sure it is much different.