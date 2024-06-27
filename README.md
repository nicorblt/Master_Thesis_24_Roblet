# Master Thesis: Spectrum of unbounded operators and applications to PDEs.

## Credit
-  Master thesis of Nicolas Roblet supervised by Romain Joly and co-supervised by Eloi Martinet.
-  Research project performed at Institut Fourier
- $5^\text{th}$ February 2024 -- $28^\text{th}$ June 2024

## Directory Structure
- **Report_Roblet.pdf:** This file is my final report.
- **/code:** This directory contains the source code used in the thesis.

## Abstract
This manuscript is the outcome of my long research project, conducted as part of the conclusion of my Ensimag engineering training and the preparation for my MSIAM double university degree.

The first section of this document outlines the context and objectives of this master thesis, thus clarifying the expectations and the chosen direction for this study.
After this introduction, we delve into the core of the work accomplished.

All the studies conducted during this master thesis are based on the fundamental result of the diagonalization of the Laplacian with homogeneous Dirichlet boundary conditions.
The second section first explores the asymptotic behavior of eigenvalues through the demonstration of Weyl's law.
We then address an optimization problem by demonstrating that the domain minimizing the fundamental eigenvalue of the Laplacian at constant volume is the ball, according to the Faber-Krahn inequality theorem.

The final section presents the numerical results obtained.
The first objective was to develop an algorithm capable of taking any domain in $\mathbb{R}^2$ as input and calculating the associated Laplacian eigenvalues and eigenfunctions.
A step-by-step study is carried out; first, we solve the one-dimensional case, then extend the reasoning to the two-dimensional rectangle, and finally, we generalize it to any domain in $\mathbb{R}^2$.
From this algorithm, we design a numerical optimization procedure to establish the Faber-Krahn inequality and extend it to the following eigenvalues.
For this shape optimization problem, we represent the domain using the phase field method.

Following this, we numerically solve shape optimization problems.
Precisely, we develop an algorithm to numerically obtain the Faber-Krahn inequality and then extend it to the following eigenvalues.

In the appendix, we present a brief demonstration of the diagonalization result of the Laplacian.
This allows us to highlight the essential results of functional analysis necessary for our study and to include, for the sake of completeness, a sketch of the proof of the central result of our studies.

The final appendix compiles various results from measure theory, thus clarifying and formalizing the statements of the results used throughout our study.\\

## Contact
*Nicolas Roblet:* Nicolas.Roblet@grenoble-inp.org  
*Romain Joly:*  Romain.Joly@univ-grenoble-alpes.fr  
*Eloi Martinet:* Eloi.Martinet@uni-wuerzburg.de

---

