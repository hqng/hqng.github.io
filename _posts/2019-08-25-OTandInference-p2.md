---
mathjax: true
title: "Optimal Transport and Variational Inference (part 2)"
classes: wide
categories:
  - Variational Inference
toc: true
tags:
  - VAE
  - OT
excerpt: "Second part of blog series about optimal transport, Wasserstein distance and generative models, variational inference and VAE."
---

## <a name="VAE"></a> Variational Auto-encoders (VAE)

VAE is another scale-up variant of VI. It employs deep neural networks to perform large datasets of high-dimensional samples such as images. Apart from representation learning, VAE is more advanced than VI at ability of reconstructing high quality samples.

### <a name="AmortizedVI"></a> Amortized Variational Inference

In VI models, each local variable is governed by its own variational parameter, e.g. in SVI, parameter $\theta_i$ corresponds to latent variable $z_i$. To maximize ELBO, we have to optimize objective function w.r.t all variational parameters. Consequently, the larger number of parameters is, the more expensive computational cost is.

<div style="text-align: center;">
<img src="{{ '/assets/otvi/AmortizedVI.jpg' | relative_url }}" alt="Amortized VI" width="25%" /> 
</div>
<a name="Fig2.1"></a> <sub> *Fig2.1:* Graphical model of Amortized VI. Dashed line indicates variational approximation. </sub>

Amortized VI reforms SVI structure to lower the cost. In particular, it  assumes that optimal $z_i$'s can be represented as a function of $x_i$s, $z_i = f(x_i)$, i.e. $z_i$s are features of $x_i$s. Of course, local variational parameters are removed. Employing a function whose parameters are shared across all data points allows past computation to support future computation. Once the function is estimated (say, after few optimization steps), local variables obviously can be computed by passing new data points to $f(\cdot)$. This is why we name it *amortized*. Function $f(\cdot)$ implements a deep neural network called *inference network* to make a powerful predictor.

### <a name="Reparmeterize-MC"></a> Reparameterization and Monte Carlo

Reparameterization trick and Monte Carlo method are necessary for VAE to work. One challenge in VAE, and gradient-based models in general, is to compute gradient of expectation of a smooth function. A common way to achieve such gradient without analytical methods is to reparameterize variable first, then estimate gradient by Monte Carlo sampling. The former step serves two purposes, one for back-propagation's, one for reducing complexity of the latter step. (Feel free to skip this section if you are already familiar with those concepts.) <br>

Let's estimate the following gradient which later shows up in VAE:
<br>
{% raw %}
$$ \small
\begin{align}
\nabla_{\theta} \E_{q_{\theta}(z)} \left[ f(z) \right] = \nabla_{\theta} \int q_{\theta}(z)f(z)dz \label{eq2.1} \tag(2.1)
\end{align}
$$
{% endraw %}
<br>
The naive Monte Carlo gradient estimator of ($\ref{eq2.1}$) is:
<br>
{% raw %}
$$ \small
\begin{align*}
\nabla_{\theta} \E_{q_{\theta}(z)} \left[ f(z) \right] &= \E_{q_{\theta}(z)} \left[ f(z) \nabla_{q_{\theta}(z)} \log q_{\theta}(z) \right] \approx \frac{1}{L} \sum_{i=1}^{L} f(z) \nabla_{q_{\theta}(z_i)} \log q_{\theta}(z_i) \\
\text{where:} \: & L \: \text{is number of samples} \nonumber \\
& z_i \sim q_{\theta}(z)
\end{align*}
$$
{% endraw %}
This often results in very high variance estimate and impractical [Blei *et al.*, 2012](https://arxiv.org/abs/1312.6114). Fortunately, reparameterization trick can resolve the problem. <br>
<br>
The idea of reparameterization is to transform one distribution into another form by additive/multiplicative location-scale transformations, these are basically [co-ordinate transformations](http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/). This way, we can express diverse and flexible class of distributions in combination of multiple simpler terms. <br>
<br>
We illustrate normal distribution case since it is widely used in machine learning and also appears in VAE. Given variable $z$ drawn from normal distribution and standard Gaussian noise {% raw %} $ \varepsilon $ {% endraw %}, {% raw %} $z$ {% endraw %} can be reparameterized by following transformation:
<br>
{% raw %}
$$ \small
\begin{align}
& z = \mu + \varepsilon \sigma \label{eq2.2} \\
\text{where:} \: & z \sim \mathcal{N}(z ; \mu, \sigma^2) \nonumber \\
& \varepsilon \sim \mathcal{N}(\varepsilon ; 0, 1) \nonumber
\end{align}
$$
{% endraw %}
<br>
If high dimensional space:
<br>
{% raw %}
$$ \small
\begin{align*}
& z = \mu + \varepsilon \odot \Sigma  \\
\text{where:} \: & z \sim \mathcal{N}(z; \mu, \Sigma \Sigma^{\top}) \\
& \varepsilon \sim \mathcal{N}(\varepsilon ; 0, 1) \\
& \odot \: \text{is point-wise product}
\end{align*}
$$
{% endraw %}
From now on, ($\ref{eq2.2}$) is used for referring both cases unless stated otherwise. <br>
<br>
With the transformation from distribution $q(\epsilon)$ to $q_{\theta}(z)$, [the probability contained in a differential area must be invariant under change of variables](https://en.wikipedia.org/wiki/Probability_density_function#Dependent_variables_and_change_of_variables), i.e. {% raw %} $ \abs{q_{\theta}(z)dz} = \abs{q(\varepsilon) d \varepsilon} $ {% endraw %}. Together with ($\ref{eq2.1}$), we have:
<br>
{% raw %}
$$ \small
\begin{alignat}{2}
& \nabla_{\theta} \E_{q_{\theta}(z)} \left[ f(z) \right] &&= \: \nabla_{\theta} \int q_{\theta}(z)f(z)dz \nonumber \\
=& \:  \nabla_{\theta} \int q(\varepsilon) f(z) d \varepsilon &&= \: \nabla_{\theta} \int q(\varepsilon) f(g(\varepsilon, \theta)) d \varepsilon \nonumber \\
=& \: \nabla_{\theta} \E_{q(\varepsilon)} \left[ f(g(\varepsilon, \theta)) \right] &&= \: \E_{q(\varepsilon)} \left[ \nabla_{\theta} f(g(\varepsilon, \theta)) \right] \label{eq2.3}
\end{alignat}
$$
{% endraw %}
Here $\theta$ is the set of parameters and $g(\varepsilon, \theta)$ is the transformation. $q_{\theta}(z)$ and $q(\varepsilon)$ are density functions of distribution of $z$ and $\varepsilon$ respectively. For instance, when $z$ has normal distribution, $\theta$ would be $\{\mu, \sigma \}$ and $g(\varepsilon, \theta)$ would be equation (\ref{eq2.2}). <br>
<br>
Gradient in ($\ref{eq2.3}$) now can be acquired using Monte Carlo estimation. Monte Carlo method allows us to estimate result of certain tasks by performing deterministic computation on large number of inputs that are sampled from a probability distribution on pre-defined domain. It eases the worry of analytically computing intractable quantity. For integral task, it is simple and straightforward:
<br>
{% raw %}
$$ \small
\begin{align}
 \E_{q(z)} \left[ f(z) \right] &= \int f(z) q(z) dz \approx \frac{1}{L} \sum_{l=1}^{L} f(z_l) \label{eq2.4} \\
\text{where:} \: & z_l \sim q(z) \: \text{for} \: l=1,2,\dots,L \nonumber
\end{align}
$$
{% endraw %}
The larger number of samples is, the more accurate estimation is. <br>

From ($\ref{eq2.3}$) and ($\ref{eq2.4}$):
<br>
{% raw %}
$$ \small
\begin{align}
& \E_{q_{\theta}(z)} \left[ f(z) \right] \approx \frac{1}{L} \sum_{i=1}^{L} f(g(\varepsilon_l, \theta)) \nonumber \\
& \nabla_{\theta} \E_{q_{\theta}(z)} \left[ f(z) \right] = \E_{q(\varepsilon)} \left[ \nabla_{\theta} f(g(\varepsilon, \theta)) \right]  \approx \frac{1}{L} \sum_{l=1}^{L} \left[ \nabla_{\theta} f(g(\varepsilon_l, \theta)) \right] \label{eq2.5} \\
& \text{where:} \: \varepsilon_l \sim q(\varepsilon) \nonumber
\end{align}
$$
{% endraw %}
Sampling $\varepsilon$ clearly is easier than sampling $z$ directly, the problem ($\ref{eq2.1}$) turns out to be feasible.

### <a name="VAEmodel"> VAE

VAE adopts SVI and Amortized VI to make a powerful generative model. The term "generative" bases on the fact that VAE employs a neural network as *generative network* alongside mentioned *inference network*. 
For simplicity, we only study VAE in setting of deep latent Gaussian model, i.e. hidden variable $z$ has (parameterized) normal distribution. Other settings which are less common can be found at [Kingma's Thesis](), [Kingma and Welling, 2014](https://arxiv.org/abs/1312.6114).

<html>
<style>
* {
  box-sizing: border-box;
}

.column {
  float: left;
  width: 33.33%;
  padding: 5px;
}

/* Clearfix (clear floats) */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>

<div class="row" style="text-align: center;">
  <div class="column">
    <img src="{{ '/assets/otvi/VAE.jpg' | relative_url }}" alt="VAE" style="width: 100%;"/>
    <figcaption>fig2.2a Grapical model</figcaption>
  </div>
  <div class="column">
    <img src="{{ '/assets/otvi/VAEnet.jpg' | relative_url }}" alt="VAEnet" style="width: 100%;">
	<figcaption>fig 2.2b Neural networks</figcaption>
  </div>
</div>
</html>

<a name="Fig2.1"></a> <sub>(a) Fig \ref{fig2.2a} shows probabilistic VAE model. Dashed lines indicate variational approximation, solid lines present generative model. $\phiparam$ is parameters of variational distribution $q_{\phiparam}(z | x)$. $\thetaparam$ is parameter of generative model $p(z) p_{\thetaparam}(x | z) $. (b) Fig \ref{fig2.2b} presents VAE deep learning model. $q_{\phiparam}(z | x)$ and $p_{\thetaparam}(x | z)$ are replaced by neural networks.</sub>

Figure (\ref{fig2.2}) demonstrates VAE in two perspectives: (a) graphical model and (b) deep learning model. Inference model with variational distribution $q_{\phiparam}(z | x)$ and generative model $p(z) p_{\thetaparam}(x | z)$ are performed by encoder network and decoder network respectively. The variational parameters $\phiparam$ and generative model's parameters $\thetaparam $ are simultaneously optimized. While VI considers a set of data points and a set of latent variables (section \ref{VI}), VAE can take a single data point as input thanks to \textit{amortized} setting. \\ %i.e. given a single observation $x_i$, we have $z_i \sim q_{\phiparam}(z | x=x_i) = q_{\phiparam}(z | x_i)$. \\

Similar to (\ref{eq1.4}) or (\ref{eq1.4a}), we can come up with objective function of VAE. Recall that out data points are i.i.d, the marginal log-likelihood is $\log p(\x) = \sum_{i=1}^{N} \log p(x_i)$. Therefore, we only concern about a single observation:
\begin{align}
\log p(x) &= \E_{z \sim q_{\phiparam} (z|x)} \left[ \log p(x) \right] \nonumber \\
&= \E_{q_{\phiparam} (z|x)} \left[ \log \frac{p_{\thetaparam}(x, z)}{p(z|x)} \right] \nonumber \\
&= \E_{q_{\phiparam} (z|x)} \left[ \log \frac{p_{\thetaparam}(x, z) q_{\phiparam}(z|x) }{p(z|x) q_{\phiparam}(z|x)} \right] \nonumber \\
&= \E_{q_{\phiparam} (z|x)} \left[ \frac{q_{\phiparam}(z|x) }{p(z|x) } \right] + \E_{q_{\phiparam} (z|x)} \left[ \log p_{\thetaparam}(x, z) - \log q_{\phiparam}(z|x) \right] \nonumber \\
&= \text{KL} \left( q_{\phiparam}(z|x) \parallel p(z|x) \right) + \E_{q_{\phiparam} (z|x)} \left[ \log p_{\thetaparam}(x, z) - \log q_{\phiparam}(z|x) \right] \nonumber \\
&= \text{KL} \left( q_{\phiparam}(z|x) \parallel p(z|x) \right) + \E_{q_{\phiparam} (z|x)} \left[ \log p_{\thetaparam}(x| z) + \log p(z) - \log q_{\phiparam}(z|x) \right] \nonumber \\
&= \text{KL} \left( q_{\phiparam}(z|x) \parallel p(z|x) \right) + \E_{q_{\phiparam} (z|x)} \left[ \log p_{\thetaparam}(x|z) \right] - \text{KL}\left( q_{\phiparam}(z|x) \parallel p(z) \right)  \label{eq2.6}
\end{align}
\begin{align}
\implies \log p(x) - \text{KL} \left( q_{\phiparam}(z|x) \parallel p(z|x) \right) &= \underbrace{ -\text{KL}\left( q_{\phiparam}(z|x) \parallel p(z) \right) + \E_{q_{\phiparam} (z|x)} \left[ \log p_{\thetaparam}(x|z) \right] }_{\ell} \tag{2.6a} \label{eq2.6a} 
\end{align}

Minimizing KL divergence between variational posterior and true posterior equivalents to maximizing ELBO $\ell$. The variational lower bound of a single data point $x_i$:
\begin{align}
\ell_i (\phiparam, \thetaparam) = - \text{KL}\left( q_{\phiparam}(z|x_i) \parallel p(z) \right) + \E_{q_{\phiparam} (z|x_i)} \left[ \log p_{\thetaparam}(x_i|z) \right] \label{eq2.7}
\end{align}
The objective function on entire data set should be:
\begin{align}
\mathcal{L} &= \sum_{i=1}^{N} \ell_i (\phiparam, \thetaparam) = - \sum_{i=1}^{N} \text{KL} \left( q_{\phiparam}(z|x_i) \parallel p(z) \right) + \sum_{i=1}^{N} \E_{q_{\phiparam} (z|x_i)} \left[ \log p_{\thetaparam} (x_i | z)  \right] \nonumber \\
&= \E_{x \sim p(x)} \left[ - \text{KL}\left( q_{\phiparam}(z|x) \parallel p(z)  \right) \right] + \E_{x \sim p(x)} \left[ \E_{q_{\phiparam} (z|x)} \left[ \log p_{\thetaparam} (x | z)  \right]  \right] \label{eq2.8}
\end{align}

The quantity $ \text{KL}\left( q_{\phiparam}(z|x_i) \parallel p(z) \right) $ can be integrated analytically under certain assumption. Let's consider our deep latent Gaussian model:
\begin{align}
p(z) &= \mathcal{N} \left(z; 0, \mathbb{I} \right) \nonumber \\
q_{\phiparam}(z | x) &= \mathcal{N}  \left(z; \mu(x), \sigma^2(x) \mathbb{I} \right) \nonumber \\
\text{where:} & \: \mu, \sigma \: \text{are functions of} \: x \nonumber
\end{align}
We have:
\begin{align}
\text{KL}\left( q_{\phiparam}(z|x) \parallel p(z) \right) &= \E_{q_{\phiparam} (z | x)} \left[ \log q_{\phiparam} (z | x) - \log p(z) \right] \nonumber \\
&= \int q_{\phiparam}(z|x) \log q_{\phiparam}(z|x)dz - \int q_{\phiparam}(z|x) \log p(z)dz \label{eq2.9}
\end{align}
Under Gaussian assumption, integrals in (\ref{eq2.9}) can be analytically computed:
\begin{align}
\int q_{\phiparam}(z|x) \log q_{\phiparam}(z|x)dz &= \int \mathcal{N} (z; \mu, \sigma^2 \mathbb{I}) \log \mathcal{N} (z; \mu, \sigma^2 \mathbb{I}) dz \nonumber \\
&= - \frac{D}{2} \log (2\pi) - \frac{1}{2} \sum_{d=1}^{D} (1 + \log \sigma_{d}^2) \tag{2.10a} \label{eq2.10a}
\end{align}
and:
\begin{align}
\int q_{\phiparam}(z|x) \log p(z)dz &= \int \mathcal{N} (z; \mu, \sigma^2 \mathbb{I}) \log \mathcal{N} (z; 0, \mathbb{I}) dz \nonumber \\
&= - \frac{D}{2} \log (2\pi) - \frac{1}{2} \sum_{d=1}^{D} (\mu_d^2 + \sigma_{d}^2) \tag{2.10b} \label{eq2.10b} \\
\text{where:} \: D \: &\text{is dimensionality of} \; z \nonumber
\end{align}
Hence:
\begin{align}
- \text{KL}\left( q_{\phiparam}(z|x_i) \parallel p(z) \right) 
= \frac{1}{2} \sum_{d=1}^{D} \left[1 + \log (\sigma_{d}^2 (x_i) )- \mu_{d}^2(x_i) - \sigma_{d}^2 (x_i) \right] \label{eq2.10}
\end{align}

The term $\E_{q_{\phiparam} (z|x_i)} \left[ \log p_{\thetaparam}(x_i|z) \right] $ is more tricky because we want both its (estimated) value and gradient w.r.t $\phiparam$. As we discuss in section \ref{Reparmeterize-MC}, using directly Monte Carlo on original variable gives high variance estimator of gradient. We therefore need the reparameterization trick. Instead of sampling $z$ from $q_{\phiparam} (z|x) = \mathcal{N} (z; \mu(x), \sigma^2(x) \mathbb{I} )$, we sample $z$ as below:
\begin{align}
&z = g(\varepsilon, \mu, \sigma) = \mu (x) + \sigma (x) \odot \varepsilon \nonumber \\
&\text{where:} \: \varepsilon \sim \mathcal{N} (0, \mathbb{I}) \nonumber
\end{align}
From (\ref{eq2.5}):
\begin{align}
& \E_{q_{\phiparam} (z|x_i)} \left[ p_{\thetaparam} (x_i|z) \right] \approx \frac{1}{L} \sum_{l=1}^{L} \log p_{\thetaparam} (x_i | g(\varepsilon_{l}, \mu_{i}, \sigma_{i} )) \nonumber \\
& \nabla_{\phiparam} \E_{q_{\phiparam} (z|x_i)} \left[ p_{\thetaparam} (x_i|z) \right] \approx 
\frac{1}{L} \sum_{l=1}^{L} \left[ \nabla_{\phi} \log p_{\thetaparam} (x_i | g(\varepsilon_{l}, \mu_{i}, \sigma_{i} )) \right] \label{eq2.11} \\
\text{where:} & \: \varepsilon_{l} \sim \mathcal{N} (0, \mathbb{I}) \nonumber
\end{align}
One combines (\ref{eq2.10}) and (\ref{eq2.11}) to get estimate of ELBO:
\begin{align}
\ell_i \approx 
\frac{1}{2} \sum_{d=1}^{D} \left[1 + \log (\sigma_{d}^2 (x_i) )- \mu_{d}^2(x_i) - \sigma_{d}^2 (x_i) \right] + 
\frac{1}{L} \sum_{l=1}^{L} \log p_{\thetaparam} (x_i | g(\varepsilon_{l}, \mu_{i}, \sigma_{i} )) \label{eq2.12}
\end{align}
Finally, objective function of VAE:
\begin{align}
\underset{\phiparam, \thetaparam}{\max} \sum_{i=1}^{N} \left( \frac{1}{2} \sum_{d=1}^{D} \left[1 + \log (\sigma_{d}^2 (x_i) )- \mu_{d}^2(x_i) - \sigma_{d}^2 (x_i) \right] \right) + 
\sum_{i=1}^{N} \left( \frac{1}{L} \sum_{l=1}^{L} \log p_{\thetaparam} (x_i | g(\varepsilon_{l}, \mu_{i}, \sigma_{i} ))  \right)
\end{align}
The first term is regularization, the second term is reconstruction cost. While regularization forces the model not to learn trivial latent space, reconstruction ensures the model outputs high quality samples that is close to input. 