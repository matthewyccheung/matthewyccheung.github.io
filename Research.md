---
layout: page
title: Research
permalink: /Research/
---

<h4> Research Statement </h4>

I am interested in continuing research in the field of Computational Imaging Systems, specifically in combining Optics and Numerical Algorithms for applications in microscopy, photography and computing. In microscopy and in photography, I am interested in making the invisible visible by jointly developing new hardware and software. In computing, I am interested in developing novel optical computing techniques. Ultimately, capturing images and processing images can be seamless, requiring no transistor and be completed in one system. <br> <br>

<h4> Deep Diffractive Neural Networks </h4>

<div style="text-align: center"><img src="/images/diff_analog.png" style="width: 50%; height: 50%"/></div>
<br>

Recently, a novel method to perform deep learning using using all optical hardware has been demonstrated ( [Lin et al. 2018](https://science.sciencemag.org/content/361/6406/1004) ). This design allows the forward propagation step to be computed at the speed of light and in parallel

However, there are several issues with the current State-of-the-art design.
1. Deep Learning must have nonlinear activation functions: the paper does not demonstrate nonlinearity. Optical Phenomena is nonlinear.
2. The large size makes scaling to larger and larger arrays difficult
3. It is non-programmable and susceptible to defects in fabrication (3D printing), making it difficult to explore other learning techniques (i.e. Reinforcement Learning, Transfer Learning)
4. It is costly: terahertz source was used to increase scattering angle

<br>
<h4> Current Research Project </h4>

<div style="text-align: center"><img src="/images/nin.jpg" style="width: 100%; height: 100%"/></div>
<br>

To address these issues, we designed a programmable and compact solution using accessible off-the-shelf components and visible light. The important theoretical ideas were:

1. To solve nonlinear property: found cheap off-the-shelf optical components with nonlinear properties
2. To build programmable array: component allowed implementation of many different architectures
3. To solve scalability issue: component was cheap and small
<br> <br>

<h4> Results so far </h4>

I built a Deep Learning Physics Simulation with an existing Deep Learning framework (Tensorflow) and am working on finding the analogs between Deep Learning architectures and optical systems to improve my accuracy. I trained the network to 80+% accuracy on the MNIST Handwritten dataset, but am in the process of improving the accuracy before submitting for publication.

The current challenges of my project are:
1. Characterizing a Deep Learning environment that has never been investigated before <br>
2. FFT was used for simulating diffraction. FFT takes a long time to process large matrices. Computing power is limited. <br>

I am currently working on implementing adaptive parameter tuning to attain better models before submitting for publication.
