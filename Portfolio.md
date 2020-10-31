---
layout: page
title: Portfolio
permalink: /Portfolio/
---

I have worked on a few independent exploratory projects related to Scientific Computing during the latter half of my undergraduate education.

----

<br>
[Iterative Subspace Projection Methods for Large Scale Linear Systems and Eigenvalue Problems](https://github.com/matthewyccheung/ISPM_COMPARISONS/blob/master/ISPM_Technical_Report.pdf)

<div style="text-align: center"><img src="/images/hessenberg.JPG" style="width: 50%; height: 50%"/></div>
<br>
* Solving Ax = b quickly is important. However, solving them for large sparse matrices is inefficient and time consuming (n-cubed by gaussian elimination). 
* Iterative Subspace Projection Methods aim to solve these problems by dimension reduction. For eigenvalue problems, these methods attain approximations for eigenvalues and eigenvectors from projections of the eigenproblem to subspaces of smaller dimension, which are expanded during the course of the algorithm.
* Solving eigenvalues are important in many fields such as physics and play an important role in subfields in Computer Science such as Computer Vision

Abstract:
<br>
The Steepest Descent (SD), Conjugate Gradient (CG), Minimal Residual (MR) and Restarted GMRES methods were implemented and the convergence behaviors of these methods were studied for a variant of problems. Furthermore, the eigenvalues accuracies were evaluated for diﬀerent conﬁgurations of the Arnoldi method. The results indicate the required properties of each method are upheld (i.e methods requiring symmetric positive deﬁnite (SPD) matrices do not converge for non-SPD), the eigenvalue accuracy for the Arnoldi method increases with increasing iteration, the Ritz values from the Hessenberg matrix are good approximations for the eigenvalues, the non-reorthogonalized Arnoldi method signiﬁcantly degrades orthogonality of resulting matrices but keep the same eigenvalues, and the shifted-inverse Arnoldi iteration converges to a target faster when speciﬁed.

----
<br>
[BLAS Comparisons](https://github.com/matthewyccheung/BLAS_COMPARISONS/blob/master/BLAS_Method_Comparison_Report.ipynb)

<div style="text-align: center"><img src="/images/runtime_blas.png" style="width: 60%; height: 60%"/></div>
<br>
* The time to run a program can be modeled by: Time to run code = clock cycles running code + clock cycles waiting for memory
* The improvement rate in processor speed (Moore's Law, ~60%/yr) exceeds that of DRAM memory (~7%/yr). There is a growing latency between the state of the art proessor speed and memory access speed (~53%/yr). Therefore, the bottleneck for computing is memory access and not the number of arithmetic operations.
* Computing the matrix product of two matrices has a computational complexity of $O(n^3)$ for $n\times n$ matrices and is slow. Different algorithms have been devised for computing the matrix product for large matrices.

Abstract:
<br>
Five different algorithms for matrix multiplication were tested and compared with Floating Point Operations Per Second (FLOPS) for square matrices. These algorithms are Triple Loop, Dot Product (BLAS-1), Saxpy (BLAS-1), Matrix-Vector (BLAS-2) and Outer-Product (BLAS-2). The size of the matrix was varied and was plotted versus MFLOPS. In this case, 10 values from 10 to 500 were selected. From the collected data, the MFLOPS ranking was matrix-vector, outer product, dot product, saxpy and triple loop. The rankings were expected because BLAS-2 were theorized to be faster than BLAS-1 and both were theorzied to be faster than the triple loop due to the number of FLOPS/MemoryReferences. The curves plotted were consistent with the expected shape - as the size of the matrix gets larger and larger, the MFLOPS eventually plateaus. This will be more evident as the size of the matrix gets larger and larger.

----
<br>
[Signal Denoising using 1D Kalman Filters](https://github.com/matthewyccheung/IMAGE_DENOISING_1D_KALMAN/blob/master/kalman_filter_1d_denoising.pdf)

<div style="text-align: center"><img src="/images/kalman_image_denoise.JPG" style="width: 70%; height: 70%"/></div>
<br>
* Kalman Filters are is a recursive optimal estimator. It operates by inferring parameters from indirect, inaccurate and uncertain observations. They are popular in guidance and navigation systems such as GPS and denoising.

Abstract:
<br>
The basic concepts of the Kalman Filter is investigated through the lens of perturbing an inverse matrix and through the full derivation. The Kalman Filter uses this concept of perturbation to ”update states” without recursively solving a larger and larger least squares problem. From the numerical experiments, the Kalman Filter serves as a good denoising ﬁlter for 1D signals. The estimated process variance and the estimated measurement variance aﬀects the convergence but the initial guess of the signal does not. The lower the estimated process variance, the better the denoising. The initial guesses display the same convergence behavior after the 1st iteration. Furthermore, gaussian noise is suppressed better than exponential and uniform noise distributions. Further applications of Kalman Filters are discussed.
