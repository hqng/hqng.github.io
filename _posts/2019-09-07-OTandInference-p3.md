---
mathjax: true
title: "Optimal Transport and Variational Inference (part 3)"
classes: wide
categories:
  - Variational Inference
toc: true
tags:
  - VAE
  - OT
excerpt: "Third part of blog series about optimal transport, Wasserstein distance and generative models, variational inference and VAE."
---

## [***Part 2***](/variational%20inference/OTandInference-p2/)


## <a name="OT"></a> Optimal Transport (OT)

Although VAE has potentials in representation learning and generative models, it may suffer from two problems: (1) uninformative features, and (2) variance over-estimation in latent space. The cause of these problems is KL divergence.
<br>

*(1) Uninformative Latent Code*: previous research show that the regularization term in ([2.8](/variational%20inference/OTandInference-p2/#eq2.8)) might be too restrictive. Particularly, {% raw %} $ \E\_{x \sim p(x)} \left[ - \text{KL} \left( q_{\phiparam}(z \| x) \parallel p(z) \right) \right] $ {% endraw %} encourages $ q\_{\phiparam}(z \| x) $ to be a random sample from $p(z)$ for every $x$, and in consequence, latent variables carry less information about input data. <br>

*(2) Variance Over-Estimation in Latent Space*: VAE tends to over-fit data due to the fact that the regularization term is not strong enough compared with the reconstruction cost. As a result of over-fitting, variance of variational distribution tends toward infinity. One can put more weight on the regularization, i.e. adding coefficient $\beta > 1$ to {% raw %} $ \E\_{x \sim p(x)} \left[ - \text{KL}\left( q_{\phiparam}(z \| x) \parallel p(z)  \right) \right] $ {% endraw %}, but it comes back to problem (1).
<br>

For more intellectual analysis on these drawbacks, one can check out [Info-VAE](https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/). Additionally, KL divergence itself has disadvantages. It is troublesome when comparing distributions that are extremely different. For example, consider 2 distributions $p(x)$ and $q(x)$ in figure \ref{fig3.1}, their masses are distributed in disparate shapes, each assigns zero probability to different families of sets

<div style="text-align: center;">
<img src="{{ '/assets/otvi/KLdrawback.png' | relative_url }}" alt="Amortized VI" width="40%" /> 
</div>

<div style="text-align: center;">
<a name="fig3.1"></a> <sub> <i>Fig3.1: Example of 2 distributions that have drastically different masses.</i> </sub>
</div>
<br>

In order to get $\text{KL} ( p \parallel q) = \E_{x \sim p(x)} \left[ \log \frac{p(x)}{q(x)} \right] $, we have to compute ratio $ \frac{p(x)}{q(x)}$ for all the points, but $q(x)$ doesn't even have density with respect to ambient space (thin line connects masses in figure [3.1](#fig3.1)). If we are interested in $\text{KL} ( q \parallel p) = \E_{x \sim q(x)} \left[ \log \frac{q(x)}{p(x)} \right] $, when $q(x) \rightarrow 0$ and $p(x) > 0 $, the divergence shrinks to $0$, it means KL cannot measure the difference between distribution properly. In contrast, optimal transport does have this problem.<br>

## <a name="Wasserstein"></a> OT and Wasserstein distance

Optimal transport is first introduced by Monge in 1781, Kantorovich later proposed a relaxation of the problem in early 20th century. We will revisit these mathematical formalism, then come up with Wasserstein distance, a special optimal transport cost that is widely used in recent generative models.<br>

**Monge's Problem**: Given measurable space $\Omega$; a cost function $c: \Omega \times \Omega \rightarrow \mathbb{R} $, $\mu$ and $\nu$ are 2 probability measures in $\mathcal{P}(\Omega)$. Monge's problem is to find a map $T: \Omega \rightarrow \Omega$ such that:
<br>
{% raw %}
$$ \small
\begin{align}
\inf_{T_{\#}\mu = \nu} \int_{\Omega} c(x, T(x)) d\mu (x) \label{eq3.1}
\end{align}
$$
{% endraw %}
<br>
where $T_{\\#} \mu$ is [*push-forward*](https://en.wikipedia.org/wiki/Pushforward_measure) operator, intuitively it moves entire distribution $\mu$ to $\nu$. Since $T$ does not always exist, Kantorovich consider probability couplings instead.<br>

**Kantorovich's Problem (Primal)**: Given $\mu$, $\nu$ in $\mathcal{P}(\Omega)$; a cost function $c$ on $\Omega \times \Omega$, the problem is to find a coupling $\gamma \in \Gamma$ such that:
<br>
{% raw %}
$$ \small
\begin{align}
\inf_{\gamma \in \Gamma(\mu, \nu)} \iint_{\Omega \times \Omega} c(x, y) d\gamma(x, y) \label{eq3.2}
\end{align}
$$
{% endraw %}
<br>
where $\Gamma$ is the set of probability couplings:
<br>
{% raw %}
$$ \small 
\begin{align*}
\Gamma(\mu, \nu) \coloneqq \: & \{ \gamma \in \mathcal{P}(\Omega \times \Omega) \mid \forall A, B \subset \Omega, \\
 &\gamma(A \times \Omega) = \mu(A), \\
 &\gamma(B \times \Omega) = \nu(B) \} 
\end{align*}
$$
{% endraw %}
<br>
Problem ($\ref{eq3.2}$) is primal form, it can be derived to duality formula: given 2 real-valued functions $\varphi$, $\psi$ on $\Omega$:
<br>
{% raw %}
$$ \small
\begin{align*}
(\varphi \oplus \psi) (x, y) \coloneqq \varphi (x) + \psi(y)
\end{align*}
$$
{% endraw %}
then minimum of Kantorovich's problem is equal to:

**Kantorovich's Problem (Duality)**:
<br>
{% raw %}
$$ \small
\begin{align}
\sup_{\varphi \oplus \psi \leq c } \int \varphi d \mu(x) + \int \psi d \nu (y) \label{eq3.3}
\end{align}
$$
{% endraw %}
<br>

*Proof:* <br>
We have the followed function only takes 2 values:
<br>
{% raw %}
$$ \small
	g_{\Gamma}(\gamma) = \sup_{\varphi, \psi} \left[ \int \varphi d \mu + \int \psi d \nu - \iint \varphi \oplus \psi d \gamma \right] = \left\{
	\begin{array}{lr}
		0 \: & \text{if } \gamma \in \Gamma \\
		+\infty \: & \text{otherwise}
	\end{array}
	\right.
$$
{% endraw %}

Put it into ($\ref{eq3.2}$), the problem can be transformed to:
<br>
{% raw %}
$$ \small
\begin{align*}
	\text{(\ref{eq3.2})} \Leftrightarrow & \inf_{\gamma \in \mathcal{P}_{+}(\Omega \times \Omega) } \iint c d \gamma + g_{\Gamma}(\gamma) \\
	\Leftrightarrow & \inf_{\gamma \in \mathcal{P}_{+}(\Omega \times \Omega) } \left[ \iint c d \gamma + \sup_{\varphi, \psi} \left( \int \varphi d \mu + \int \psi d \nu - \iint \varphi \oplus \psi d \gamma \right) \right] \\
	\Leftrightarrow & \inf_{\gamma \in \mathcal{P}_{+}(\Omega \times \Omega) } \sup_{\varphi, \psi} 
	\left[ \iint \left( c - \varphi \oplus \psi \right) d \gamma + \int \varphi d \mu + \int \psi d \nu \right] \\
	\Leftrightarrow &  \sup_{\varphi, \psi} \inf_{\gamma \in \mathcal{P}_{+}(\Omega \times \Omega) } 
	\left[ \iint \left( c - \varphi \oplus \psi \right) d \gamma + \int \varphi d \mu + \int \psi d \nu \right] \\
	\Leftrightarrow & \sup_{\varphi, \psi} \left[ \inf_{\gamma \in \mathcal{P}_{+}(\Omega \times \Omega) } \iint \left( c - \varphi \oplus \psi \right) d \gamma + \int \varphi d \mu + \int \psi d \nu \right] \tag{$\star$}
\end{align*}
$$
{% endraw %}

But we know that:
<br>
{% raw %}
$$ \small
	\inf_{\gamma \in \mathcal{P}_{+}(\Omega \times \Omega) } \iint \left( c - \varphi \oplus \psi \right) d \gamma = \left\{
	\begin{array}{lr} 
      0 \: & \text{if } c - \varphi \oplus \psi \geq 0 \\
      -\infty \: & \text{otherwise}
    \end{array}
	\right.
$$
{% endraw %}

Hence:
<br>
{% raw %}
$$ \small
\begin{align*}
	(\star) \Leftrightarrow \sup_{\varphi \oplus \psi \leq c } \int \varphi d \mu + \int \psi d \nu
\end{align*}
$$
{% endraw %} <p style="text-align:right">&#8718;</p>

When cost function $c(x, y)$ is a metric $D^p(x,y)$, optimal transport cost is simplified to $p$*-Wasserstein distance* $W_p$:

**$p$-Wasserstein distance**:
<br>
{% raw %}
$$ \small
\begin{align}
W_p (\mu, \nu) & \coloneqq \left( \inf_{\gamma \in \Gamma(\mu, \nu)} \iint D^p(x,y) d \gamma(x, y) \right)^{1/p} \label{eq3.4} \\
W_p^p (\mu, \nu) & \coloneqq \sup_{\varphi (x) + \psi (y) \leq D^p(x,y)} \int \varphi d \mu + \int \psi d \nu \label{eq3.5}
\end{align}
$$
{% endraw %}

Equations ($\ref{eq3.4}$) and ($\ref{eq3.5}$) are primal and duality forms respectively.<br>

Assume $\varphi$ is known, we would like to find a good $\psi$ to solve ($\ref{3.5}$). Under this assumption, $\psi$ must satisfy below condition:
<br>
{% raw %}
$$ \small
\begin{align}
\psi(y) \leq & \: D^p(x,y) - \varphi(x) \: \forall x, y \nonumber \\
\Leftrightarrow \psi(y) \leq & \: \inf_{x} D^p(x,y) - \varphi(x) \eqqcolon \bar{\varphi}_x(y)  \label{eq3.6}
\end{align}
$$
{% endraw %}

The R.H.S of ($\ref{eq3.6}$) is called $D^p$-transform (of $\varphi$), of course we might exchange $\varphi$ for $\psi$ and get the $D^p$-transform of $\psi$ instead. The duality of $p$-Wasserstein now can be rewritten in semi-duality form:
<br>
{% raw %}
$$ \small
\begin{align}
W_p^p (\mu, \nu) = \sup_{\varphi} \int \varphi d \mu + \int \bar{\varphi} d \nu \label{eq3.7}
\end{align}
$$
{% endraw %}

Recall the definition of **$D^p$-concavity**: a function $\varphi (x)$ is $D^p$-concave if there exists $\phi(y)$ such that: $\varphi(x) = \bar{\phi}(x)$ (where $\varphi,\: \phi$ are "well-defined" on $\Omega$). Thus, if $\varphi$ is $D^p$-concave: $\exists \phi \: \text{s.t.} \: \varphi(x) = \bar{\phi}(x) \implies \bar{\varphi}(y) = \phi(y) \implies \bar{\bar{\varphi}}(x) = \bar{\phi}(x) = \varphi(x) $. Put the constraint into ($\ref{eq3.7}$):
<br>
{% raw %}
$$ \small
\begin{align}
W_p^p (\mu, \nu) = \sup_{\varphi \: \text{is $D^p$-concave}} \int \varphi d \mu + \int \bar{\varphi} d \nu \label{eq3.8}
\end{align}
$$
{% endraw %}

In machine learning, we often take $p=1$ and use 1-Wasserstein distance to measure the discrepancy between distributions, the duality form becomes:
<br>
{% raw %}
$$ \small
\begin{align}
\mathbf{W_1}(\mu, \nu) = \sup_{\varphi \: \text{is 1-Lipschitz}} \int_{\Omega} \varphi (d\mu - d\nu) \label{eq3.9}
\end{align}
$$
{% endraw %}

To arrive ($\ref{eq3.9}$), we must show that: $p=1$ and $\varphi$ is concave $\Leftrightarrow$ $\bar{\varphi} = - \varphi$ and $\varphi$ is 1-Lipshitz

*Proof*: <br>
Define $\bar{\varphi}_x(y) \coloneqq D(x,y) - \varphi(x)$, obviously:
<br>
$$\bar{\varphi}_x(y) - \bar{\varphi}_x(y^{\prime}) = D(x,y) - D(x,y^{\prime}) \leq D(y,y^{\prime}) \implies \varphi_x(y) \: \text{is 1-Lipschitz}$$
<br>
$$\implies \bar{\varphi}(y) = \inf_{x}\bar{\varphi}_x(y) \: \text{is 1-Lipschitz}$$
<br>
$$ \implies \bar{\varphi}(y) - \bar{\varphi}(x) \leq D(x,y)$ $\implies -\bar{\varphi}(x) \leq D(x,y) - \bar{\varphi}(y)$$
<br>
$$\implies -\bar{\varphi}(x) \leq \inf_{y} D(x,y) - \bar{\varphi}(y)$$
<br>
$$\implies -\bar{\varphi}(x) \leq \inf_{y} D(x,y) - \bar{\varphi}(y) \leq -\bar{\varphi}(x)$$
<br>
$$\implies -\bar{\varphi}(x) \leq \bar{\bar{\varphi}}(x) \leq -\bar{\varphi}(x) \implies \bar{\varphi}(x) = -\bar{\bar{\varphi}}(x) = -\varphi(x)$$ <p style="text-align:right">&#8718;</p>

One interested in detailed proofs can refer to ([Gabriel Peyre and Marco Cuturi, 2018](https://arxiv.org/abs/1803.00567)) and [Cuturi's talk](https://www.youtube.com/watch?v=1ZiP_7kmIoc&t=1500s).
Side note: Discriminator of Wasserstein GAN serves as function $\varphi$ of semi-duality form ([Aude Genevay *et al,*, 2017](https://arxiv.org/abs/1706.01807)), 1-Lipschitz constraint is fulfilled by weight-clipping ([Martin Arjovsky *et al.*, 2017](https://arxiv.org/abs/1701.07875)) or penalizing gradient (WGAN-GP, [Ishaan Gulrajani *et al.*, 2017](https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans)).

## Empirical Wasserstein distance

We have briefly covered basics of optimal transport. Solving OT is rather problematic except for certain cases, e.g. univariate or Gaussian measures. Our primary objective is to efficiently compute Wasserstein distance on empirical measures which appear in probabilistic models frequently.<br>

We consider 2 measures $\mu=\sum_{i=1}^{n} a_{i} \delta_{x_{i}}$ and $\nu=\sum_{j=1}^{m} b_{j} \delta_{y_{j}}$ where $\delta_{x_{i}}$, $\delta_{y_{j}}$ are Dirac functions at $x_i$, $y_j$ respectively. In this particular case, cost function and coupling set are specified as:
<br>
{% raw %}
$$ \small
\begin{align}
M_{X Y} \coloneqq& \left[D\left(x_{i}, y_{j}\right)^{p}\right]_{i j} \nonumber \\
U(a, b) \coloneqq& \left\{P \in \mathbb{R}_{+}^{n \times m} | P \mathbf{1}_{m}=a, P^{T} \mathbf{1}_{n}=b\right\} \nonumber
\end{align}
$$
{% endraw %}

We then can substitute Frobenius inner product for integral in OT's primal form:
<br>
{% raw %}
$$ \small
\begin{align}
& W_{p}^{p}(\mu, \nu)=\min _{P \in U(a, b)}\left\langle P, M_{X Y}\right\rangle \label{eq3.10} \\
\text{where:} \: & \left\langle \cdot, \cdot \right\rangle \: \text{is \href{https://en.wikipedia.org/wiki/Frobenius_inner_product}{Frobenius inner product}} \nonumber
\end{align}
$$
{% endraw %}

Dual form:
<br>
{% raw %}
$$ \small
\begin{align}
W_{p}^{p}(\mu, \nu)=\max _{\alpha \in \mathbb{R}^{n}, \beta \in \mathbb{R}^{m}} \alpha^{T} a+\beta^{T} b \label{eq3.11}
\end{align}
$$
{% endraw %}

One challenge is that solution of ($\ref{eq3.10}$),($\ref{eq3.11}$) is unstable and not always unique ([Cuturi's, 2019](https://www.youtube.com/watch?v=1ZiP_7kmIoc&t=1500s)). Additionally, $W_p^p$ is not differentiable, making training models by stochastic gradient optimization less feasible. Fortunately, entropic regularization that measures the level of uncertainty in a probability distribution can overcome these disadvantages:<br>

**Entropic Regularization**:
For joint distribution $P(x, y)$ (in this section, we only concern about discrete distribution unless stated otherwise):
<br>
{% raw %}
$$ \small
\begin{align*}
\mathcal{H}(P) \coloneqq - \sum_{i} \sum_{j} P(x_i,y_j) \log P(x_i,y_j)
\end{align*}
$$
{% endraw %}

For particular $P \in U(a,b)$ : $\mathcal{H}(P) = -\sum_{i,j=1}^{n,m} P(x_i,y_j) \left(\log P(x_i,y_j) -1 \right) = \sum_{i,j=1}^{n,m} P_{ij} \left(\log P_{ij} -1 \right) $
<br>

**Regularized Wasserstein**:
<br>
{% raw %}
$$ \small
\begin{align}
& W_{\epsilon}(\mu, \nu) = \min _{P \in U(a, b)} \left\langle P, M_{X Y}\right\rangle - \epsilon \mathcal{H}(P) \label{eq3.12} \\
\text{where:} \: & \epsilon \geq 0 \: \text{is regularization coeficient} \nonumber
\end{align}
$$
{% endraw %}

Strong concavity property of entropic regularization ensures the solution of ($\ref{eq3.12}$) is unique. Moreover, it can achieve a differentiable solution using Sinkhorn's algorithm. To come up with Sinkhorn iteration, we need an additional proposition.<br>

**Prop.** If $P_{\epsilon} \coloneqq \arg\min_{P \in U(a, b)} \left\langle P, M_{X Y}\right\rangle - \epsilon \mathcal{H}(P) $ then: $ \exists ! u \in \mathbb{R}_{+}^{n}, v \in \mathbb{R}_{+}^{m} $ such that: 
{% raw %} $$ P_{\epsilon}=\operatorname{diag}(u) K \operatorname{diag}(v) \: \text{with} \: K \coloneqq e^{-M_{X Y} / \epsilon} $$ {% endraw %}

*Proof*: <br>
We have:
{% raw %} 
$$ \small
L(P, \alpha, \beta) = \sum_{i j} P_{i j} M_{i j} + \epsilon P_{i j}\left(\log P_{i j}-1\right)+\alpha^{T}(P \mathbf{1}-a)+\beta^{T}\left(P^{T} \mathbf{1}-b\right) 
$$
{% endraw %}
<br>
{% raw %} 
$$ \small 
\frac{\partial L}{\partial P_{ij}} = M_{i j} + \epsilon \log P_{ij} + \alpha_i + \beta_j 
$$ 
{% endraw %}
<br>
Set this partial derivative equal to $0$, we get:
<br>
{% raw %} 
$$ \small 
P_{i j}=e^{\frac{\alpha_{i}}{\epsilon}} e^{-\frac{M_{i j}}{\epsilon}} e^{\frac{\beta_{j}}{\epsilon}}=u_{i} K_{i j} v_{j} 
$$ 
{% endraw %}
<br>
hence:
<br>
{% raw %} 
$$ \small
P_{\epsilon} \in U(a, b) \Leftrightarrow \left\{ 
	\begin{array}{ll}
		{\operatorname{diag}(u) K \operatorname{diag}(v) \mathbf{1}_{m}} & {=a} \\ 
		{\operatorname{diag}(v) K^{T} \operatorname{diag}(u) \mathbf{1}_{n}} & {=b}
	\end{array}
	\right. 
$$ 
{% endraw %}
<br>
{% raw %} 
$$ \small
\implies P_{\epsilon} \in U(a, b) \Leftrightarrow \left\{ 
	\begin{array}{ll}
		{\operatorname{diag}(u) K v} & {=a} \\ 
		{\operatorname{diag}(v) K^{T} u } & {=b} 
	\end{array} 
	\right. 
$$
{% endraw %}
<br>
{% raw %} 
$$ \small
\implies P_{\epsilon} \in U(a, b) \Leftrightarrow \left\{ 
	\begin{array}{ll}
		{u \odot K v} & {=a} \\ 
		{v \odot K^{T} u } & {=b}
	\end{array} 
	\right. 
$$
{% endraw %}
<br>
{% raw %} 
$$ \small
\implies \left\{
	\begin{array}{ll}
		u &= a / Kv \\
		v &= b / K^Tu
	\end{array}
	\right. 
$$
{% endraw %} <p style="text-align:right">&#8718;</p>

The above prop. suggests that if there exists a solution for regularized Wasserstein, it is unique and possibly computed once $u, v$ are available. As seen in the proof, these quantities can be approximated by repeating the last equation, in detail:<br> 

**Sinkhorn's algorithm**: Input $M_{XY}, \epsilon, a, b$. Initialize $u, v$. Calculate $K = e^{-M_{XY}/\epsilon}$. Repeat until convergence:
<br>
{% raw %}
$$ \small
\begin{align}
	\begin{array}{ll}
		u &= a / Kv \\
		v &= b / K^Tu
	\end{array} \label{eq3.13}
\end{align}
$$
{% endraw %}
Clearly, Sinkhorn iteration is differentiable.<br>

Sinkhorn's algorithm involves with a number of other measures in OT but we will skip them since it is  irrelevant to the next section, Wasserstein distance in VI. One last thing to remember is that when regularization coefficient $\epsilon$ tends to infinity, Sinkhorn's distance turns into Maximum Mean Discrepancy (MMD) distance. <br>


## [***Part 4***](/variational%20inference/OTandInference-p4/)