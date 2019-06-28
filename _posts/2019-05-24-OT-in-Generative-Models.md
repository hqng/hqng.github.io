---
mathjax: true
title: "Optimal Transport and VI"
categories:
  - Variational Inference
toc: true
tags:
  - VAE
  - OT
---

In N-dimensional simplex noise, the squared kernel summation radius $r^2$ is $\frac 1 2$

$$ R_{\mu \nu} - \frac{1} {2}Rg_{\mu \nu} + \Lambda g_{\mu \nu} = \frac{8\pi G} {c^4}T_{\mu \nu} $$

\begin{align}
  \max &\: \text{ELBO} - \sum_{j=1}^{M} \lambda_j \int_{z_j}q(z_j)dz_j \label{eq1.13}
\end{align}