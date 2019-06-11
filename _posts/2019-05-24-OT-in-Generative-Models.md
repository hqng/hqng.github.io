---
layout: post
mathjax: true
comments: false
title: "Optimal Transport and VI"
categories:
  - Variational Inference
tags:
  - vae
  - OT
---

---
abstract: |
    Many recent advances in generative models have been adapted from
    Bayesian probabilistic models. VAE that enhances VI, a probabilistic
    model by Deep Learning is prominent as a powerful tool for
    representation learning but stable in training. The main idea of VAE is
    to minimize KL divergence between parameterized posterior and true
    posterior with respect to variational family. Naturally, several works
    have focused on different probability measures that can be integrated to
    VI to improve representation learning capacity and generative samples
    quality. Among these approaches, models that use divergences brought
    from Optimal Transport theory (OT) are experimentally more powerful,
    flexible yet still stable. This short article reviews recent notable
    research in VI that involve with OT measurement.
author:
- hqng
bibliography:
- 'bib/reference.bib'
title: Optimal Transport in Variational Inference
---

Variational Inference
=====================

We first quickly revisit VI to get general idea of how it works. Suppose
we have a set $\operatorname{\mathbf{x}}$ of $N$ observations of data
$\operatorname{\mathbf{x}}= \{ x_1, x_2, \dots, x_N \} $. The question
is what we can tell about data from these observations? To do so, we
need to acquire meaningful representation of data. Notice that raw data
are often high-dimensional, we prefer to learn its representation in
low-dimensional setting.\
Bayesian models take into account a set of $M$ latent variables
$\operatorname{\mathbf{z}}= \{ z_1, z_2, \dots, z_M\} \sim q(\operatorname{\mathbf{z}})$
draw from prior density $q(\operatorname{\mathbf{z}})$ as data
representations and relate them to the observations through likelihood
$p(\operatorname{\mathbf{x}}| \operatorname{\mathbf{z}})$. The latent
variables are in low-dimensional space. Bayesian inference interests in
the following conditional density: $$\begin{aligned}
& p(\operatorname{\mathbf{z}}| \operatorname{\mathbf{x}}) = \frac{p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}})}{p(\operatorname{\mathbf{x}})}  = \frac{p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}})}{\int p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) d \operatorname{\mathbf{z}}} \label{eq1.1} \\
\text{where:} \: & p(\operatorname{\mathbf{z}}| \operatorname{\mathbf{x}}) \: \text{is posterior} \nonumber \\
& p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) = p(\operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}) q(\operatorname{\mathbf{z}}) \: \text{is joint density of} \: \operatorname{\mathbf{x}}\: \text{and} \: \operatorname{\mathbf{z}}\nonumber \\
& p(\operatorname{\mathbf{x}}) = \int p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) d \operatorname{\mathbf{z}}\: \text{is evidence, computed by marginalizing} \: \operatorname{\mathbf{z}}\nonumber\end{aligned}$$
The posterior represents distribution of latent variables given the
observations, getting posterior is equivalent to learning data
representation.\
While $ p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) $ can be
fully observable, the integral term is computationally expensive, thus
the posterior is intractable [@doi:10.1080/01621459.2017.1285773]. VI
overcomes this difficulty by approximating intractable posterior with
simpler distribution. It introduces parameterized prior
$q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}})$ with
variational parameters
$\operatorname{\boldsymbol{\theta}}= \{\theta_1, \theta_2, ..., \theta_M \}$
and then optimize them to get the best approximation in term of KL
divergence.

Vanilla VI
----------

We now derive the optimization problem’s objective of VI. Let’s
consider: $$\begin{aligned}
& \log p(\operatorname{\mathbf{x}}) = \log \int p(\operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}) q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}}) d\operatorname{\mathbf{z}}= \log \operatorname{\mathbb{E}}_{\operatorname{\mathbf{z}}\sim q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}})} [p(\operatorname{\mathbf{x}}| \operatorname{\mathbf{z}})] \label{eq1.2}\end{aligned}$$

Since $ \log $ is concave function, by Jensen’s inequality, we have:
$$\begin{aligned}
\log \operatorname{\mathbb{E}}_{ \operatorname{\mathbf{z}}\sim q_{\operatorname{\boldsymbol{\theta}}} (z)} [ p(\operatorname{\mathbf{x}}| \operatorname{\mathbf{z}})] \geq &\operatorname{\mathbb{E}}_{\operatorname{\mathbf{z}}\sim q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}})} [ \log p(\operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}) ] \nonumber \\ 
&= \operatorname{\mathbb{E}}_{q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}})} \left[ \log \frac{ p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) }{q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}})} \right] \nonumber \\
&= \operatorname{\mathbb{E}}_{q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}})} [ \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) - \log q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}}) ] = \mathbf{ELBO} \label{eq1.3}\end{aligned}$$
The RHS quantity is ELBO - Evidence Lower BOund.\
We now show that the difference between $\log p(x)$ and ELBO is exactly
KL divergence between variational distribution, i.e. parameterized prior
$q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}})$ and
posterior: $$\begin{aligned}
\log p(\operatorname{\mathbf{x}}) - \text{ELBO} &= \log p(\operatorname{\mathbf{x}}) - \operatorname{\mathbb{E}}_{q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}})} [ \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) - \log q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}})] \nonumber \\
&= \operatorname{\mathbb{E}}_{q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}})} [\log p(\operatorname{\mathbf{x}})] - \operatorname{\mathbb{E}}_{q_{\operatorname{\boldsymbol{\theta}}}  (\operatorname{\mathbf{z}})} [ \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) - \log q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}})] \nonumber \\
&= \operatorname{\mathbb{E}}_{q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}})} [\log p(\operatorname{\mathbf{x}}) - \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) + \log q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}}) ] \nonumber \\
&= \operatorname{\mathbb{E}}_{q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}})} \left[ -\log \frac{p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}})}{p(\operatorname{\mathbf{x}})} + \log q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}}) \right] \nonumber \\
&= \operatorname{\mathbb{E}}_{q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}})} \left[ \log q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}}) - \log p(\operatorname{\mathbf{z}}| \operatorname{\mathbf{x}}) \right] \nonumber \\
&= \operatorname{\mathbb{E}}_{q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}})} \left[ \log \frac{q_{\operatorname{\boldsymbol{\theta}}} (\operatorname{\mathbf{z}})}{p(\operatorname{\mathbf{z}}| \operatorname{\mathbf{x}})} \right] = \text{KL}(q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}}) || p(\operatorname{\mathbf{z}}| \operatorname{\mathbf{x}})) \label{eq1.4} \\
\text{where:} \: \text{KL} (q || p ) \: &\text{is Kullback-Leibler divergence between} \: q \: \text{and} \: p \nonumber\end{aligned}$$

From (\[eq1.4\]), the posterior
$p(\operatorname{\mathbf{z}}| \operatorname{\mathbf{x}})$ can be
approximated by
$q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}})$ as
long as we can find a parameters’ set
$\operatorname{\boldsymbol{\theta}}$ to have
$\text{KL}(q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}}) || p(\operatorname{\mathbf{z}}| \operatorname{\mathbf{x}})) = 0$.
It’s practically impossible to achieve that perfect goal, our best wish
is to make KL divergence as small as possible. Hence, VI simply turns
intractable posterior computing task into optimization problem with
following objective: $$\begin{aligned}
\underset{\operatorname{\boldsymbol{\theta}}}{\min} \: \text{KL}(q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}}) || p(\operatorname{\mathbf{z}}| \operatorname{\mathbf{x}}))\end{aligned}$$

Note that $\log p(\operatorname{\mathbf{x}})$ is a constant quantity
w.r.t $\operatorname{\boldsymbol{\theta}}$, to minimize
$\text{KL}(q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}}) || p(\operatorname{\mathbf{z}}| \operatorname{\mathbf{x}}))$
is equivalent to maximize the ELBO. Computing ELBO can be done
analytically by restricting models to conjugate exponential family
distribution. But we’ll focus on other approach here since it directs to
VAE, that is mean field VI.

Mean Field VI (MFVI)
--------------------

As mean field theory in physics, MFVI factorizes
$q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}})$ into
$M$ factors in which each factor is governed by its own parameter and is
independent of others: $$\begin{aligned}
q_{\operatorname{\boldsymbol{\theta}}}(\operatorname{\mathbf{z}}) = \prod_{j=1}^{M} q_{\theta_j}(z_j) \label{eq1.5}\end{aligned}$$
For brevity, from now on we shorten $q_{\theta_j}(z_j)$ as $q(z_j)$ and
denote
$\operatorname{\mathbf{z}}_{-j} = \operatorname{\mathbf{z}}\setminus {z_j}$,
i.e. the latent set excluded variable $z_j$. Note that:
$$\begin{aligned}
p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) &= p(z_j, \operatorname{\mathbf{x}}| z_{-j}) q_{-j}(\operatorname{\mathbf{z}}_{-j}) \nonumber \\
&= p(z_j, \operatorname{\mathbf{x}}| z_{-j}) \prod_{i \neq j} q_{i}(z_i) \label{eq1.6} \\
\mathbb{E}_{q(\operatorname{\mathbf{z}})}\left[\log q (\operatorname{\mathbf{z}}) \right] &= \sum_{j=1}^{M} \mathbb{E}_{q(z_j)}\left[\log q(z_j) \right] \label{eq1.7}\end{aligned}$$
Hence: $$\begin{aligned}
\text{ELBO} &= \int_{\operatorname{\mathbf{z}}} \left( \prod_{i=1}^{M} q_i (z_i) \right) \log \frac{p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}})} {\prod_{k=1}^{M} q_k(z_k) }d z_1 d z_2 \dots d z_M \nonumber \\
&= \int_{\operatorname{\mathbf{z}}} \left( \prod_{i=1}^{M} q_i (z_i) \right) \left( \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) - \sum_{k=1}^{M} \log q_k(z_k) \right) d z_1 d z_2 \dots d z_M \nonumber \\
&= \int_{z_j} q(z_j) \int_{\operatorname{\mathbf{z}}_{-j }} \left( \prod_{i \neq j} q_i(z_i) \right) \left[ \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) - \sum_{k=1}^{M} \log q_k(z_k) \right) d z_1 d z_2 \dots d z_M \nonumber \\
&= \int_{z_j} q(z_j) \int_{\operatorname{\mathbf{z}}_{-j }} \left( \prod_{i \neq j} q_i(z_i) \right) \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) d z_1 d z_2 \dots d z_M \nonumber \\
& - \int_{z_j} q(z_j) \int_{\operatorname{\mathbf{z}}_{-j }} \left( \prod_{i \neq j} q_i(z_i) \right) \sum_{k=1}^{M} \log q_k(z_k) d z_1 d z_2 \dots d z_M \label{eq1.8}\end{aligned}$$

On the other hand: $$\begin{aligned}
\int_{\operatorname{\mathbf{z}}_{-j }} \left( \prod_{i \neq j} q_i(z_i) \right) \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) dz_1 \dots dz_{j-1} dz_{j+1} \dots dz_M = \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})} \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) \label{eq1.9}\end{aligned}$$

From (\[eq1.8\]) and (\[eq1.9\]): $$\begin{aligned}
\text{ELBO} &= \int_{z_j} q(z_j) \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[ \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) ] dz_j - \int_{z_j} q(z_j) \int_{\operatorname{\mathbf{z}}_{-j }} \left( \prod_{i \neq j} q_i(z_i) \right) \sum_{k=1}^{M} \log q_k(z_k) d z_1 d z_2 \dots d z_M \nonumber \\
&= \int_{z_j} q(z_j) \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[ \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) ] dz_j 
- \int_{z_j} q(z_j) \log q(z_j) \underbrace{\int_{\operatorname{\mathbf{z}}_{-j}} \left( \prod_{i \neq j}q_i(z_i) \right) dz_1 \dots dz_M }_{=1} \nonumber \\
&- \underbrace{\int_{z_j} q(z_j) dz_j }_{=1} \int_{\operatorname{\mathbf{z}}_{-j}} \left( \prod_{i \neq j} q_i(z_i) \right) \sum_{k \neq j} \log q_k (z_k) dz_1 \dots dz_{j-1} dz_{j+1} \dots dz_M \nonumber \\
&= \int_{z_j} q(z_j) \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[ \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) ] dz_j - \int_{z_j} q(z_j) \log q(z_j) dz_j \nonumber \\
&- \int_{\operatorname{\mathbf{z}}_{-j}} \left( \prod_{i \neq j} q_i(z_i) \right) \sum_{k \neq j} \log q_k(z_k) dz_1 \dots dz_{j-1} dz_{j+1} \dots dz_M \nonumber \\
&= \int_{z_j} q(z_j) \left( \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[ \log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}}) ] - \log q(z_j) \right)  dz_j + C_{-j} \label{eq1.10} \\
\text{where:} \: & C_{-j} \: \text{containts all constant quantities w.r.t} \: z_j \nonumber\end{aligned}$$

Using (\[eq1.6\]), we can have a more common result in many VI tutorial:
$$\begin{aligned}
\text{ELBO} &= \int_{\operatorname{\mathbf{z}}_j} q(z_j) \left( \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[ \log p(z_j, \operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}_{-j}) + \log q(\operatorname{\mathbf{z}}_{-j})] - \log q(z_j) \right) dz_j + C_{-j} \nonumber \\
&= \int_{z_j} q(z_j) \left( \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[\log p(z_j, \operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}_{-j})] - \log q(z_j) \right) dz_j \nonumber \\
&+ \left( \int_{z_j} q(z_j) dz_j \right) \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})} [\log q(\operatorname{\mathbf{z}}_{-j})] + C_{-j} \nonumber \\
&= \int_{z_j} q(z_j) \left( \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[\log p(z_j, \operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}_{-j})] - \log q(z_j) \right) dz_j + C_{-j}^{\prime} \label{eq1.11}\end{aligned}$$

Our aim now becomes: $$\begin{aligned}
& \underset{q(z_j)}{\max} \int_{z_j} q(z_j) \left( \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[ \log p(z_j, \operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}_{-j}) ] - \log q(z_j) \right)  dz_j + C_{-j}^{\prime} \label{eq1.12} \\
\text{s.t:} & \: \int_{z_j}q(z_j)dz_j = 1, \: \forall j \in \{1,2,\dots,M \} \nonumber\end{aligned}$$

Problem (\[eq1.12\]) can be easily solved by Lagrange multiplier:
$$\begin{aligned}
\max &\: \text{ELBO} - \sum_{j=1}^{M} \lambda_j \int_{z_j}q(z_j)dz_j \label{eq1.13}\end{aligned}$$

Taking derivative of (\[eq1.13\]) w.r.t $q(z_j)$: $$\begin{aligned}
\frac{\partial ELBO}{\partial q(z_j)} &= \frac{\partial}{\partial q(z_j)} \left[ q(z_j) 
\left( \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})} [\log p(z_j, \operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}_{-j} ) -\log q(z_j) ] \right) - \lambda_j q(z_j) \right] \nonumber \\
&= \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[\log p(z_j, \operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}_{-j}) ] - \log q(z_j) - 1 - \lambda_j \label{eq1.14}\end{aligned}$$

Set the partial derivative to $0$ to get the updating form of $q(z_j)$:
$$\begin{aligned}
{2}
& \log q(z_j) &&= \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[\log p(z_j, \operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}_{-j} )] - 1 - \lambda_j \nonumber \\
& &&= \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[\log p(z_j, \operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}_{-j} )] + const \nonumber \\
\implies & q(z_j) &&= \frac{\exp \left\{ \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[\log p(z_j, \operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}_{-j} )] \right\} }{Z_j} \nonumber \\
\implies & q(z_j) && \propto \exp \left\{ \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[\log p(z_j, \operatorname{\mathbf{x}}| \operatorname{\mathbf{z}}_{-j} )] \right\} \nonumber \\
& && \propto \exp \left\{ \operatorname{\mathbb{E}}_{q(\operatorname{\mathbf{z}}_{-j})}[\log p(\operatorname{\mathbf{x}}, \operatorname{\mathbf{z}})] \right\} \label{eq1.15} \\
& \text{where:} \: && Z_j \: \text{is a normalization constant} \nonumber\end{aligned}$$

Since $q(z_j)$ and $q(z_i)$ are independent for any
$j \neq i, \: i, j \in \{1, 2, \dots, M \}$, maximizing EBLO w.r.t
$\operatorname{\boldsymbol{\theta}}$ can be done by alternately
maximizing ELBO w.r.t $\theta_j$, i.e. $q(z_j)$ for $j=1,2,\dots,M$.
Therefore, under mean field approximation, maximum of ELBO can be
achieved by iteratively updating variational distribution of each latent
variable as (\[eq1.15\]) until convergence. This algorithm’s called
coordinate ascent.

Stochastic VI (SVI)
-------------------

Variational Auto-encoders (VAE)
===============================
