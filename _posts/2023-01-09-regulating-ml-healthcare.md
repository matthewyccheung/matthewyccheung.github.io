---
layout: single
classes: wide
author_profile: true
title: "Regulating ML for Healthcare"
show_date: true
comments: true
---


Artificial Intelligence Algorithms in Health Care: Is the Current Food and Drug Administration Regulation Sufficient? https://ai.jmir.org/2023/1/e42940/PDF
The current benchmark for approval requires companies to demonstrate good model performance on a varied data set and in a real-world setting. With no explicit definition of what constitutes reproducible standards, it is no surprise that the current FDA-approved ML algorithms vary considerably by the size of data sets and number of sites
Under the current regulation, once an algorithm is approved, its behavior will remain fixed, defeating the distinguishing advantage of many adaptive algorithms in their ability to learn throughout their life cycle. Its current inflexible state not only reduces its clinical utility but can alarmingly infringe patient safety.
Additionally, the current FDA framework does not regulate the inception of an ML algorithm. As a result, a number of algorithms have been approved, many of similar use cases with varying development sites and data sizes. This can potentially constitute an inefficient use of resources


Are these questions really unique to ML algorithms? Every medical device which computes  converts data from one state  to another, and in essence implements a function. Many parts of these systems are based on things like regression coefficients from “some” data. Or are  thresholds to decide when an alarm goes off or something is labeled tumor or not. All those algorithms are also “fixed”, based on some “data” and could also infringe safety for same reasons.

never thought about that before, and I agree.
I saw these two relevant JAMA articles last month that I shared to CCD:
Pulse Oximeters and Violation of Federal Antidiscrimination Law: https://jamanetwork.com/journals/jama/fullarticle/2800468
Clinical Algorithms, Antidiscrimination Laws, and Medical Device Regulation: https://jamanetwork.com/journals/jama/article-abstract/2800368


Biased pulse oximeters have remained in use for decades without legal or regulatory action. A recently proposed rule by U.S. Department of Health & Human Services (HHS) may make these biased devices a Violation of Federal Antidiscrimination Law.

https://jamanetwork.com/journals/jama/article-abstract/2800468?guestAccessKey=b7c6577e-eba7-4c3e-9490-d1b457a49882&utm_source=twitter&utm_medium=social_jama&utm_term=8582511759&utm_campaign=article_alert&linkId=196571615

most validation chooses a representative set and currently, the representative set is often not characterized by race/ethnicity but by clinical measures. What is the right representative set? That is a golden question

1) during clinical trials, how should we determine representative sample (note all trials are cost constrained)?, 2) once the device is in use, how do understand when it is being used out of domain? and 3) should device algorithms, once in use, allowed to learn and adapt?