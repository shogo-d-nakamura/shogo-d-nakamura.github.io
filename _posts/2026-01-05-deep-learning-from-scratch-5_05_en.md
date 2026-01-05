---
layout: post
title: Notes on Deep Learning from Scratch 5 (Steps 5)
date: 2026-01-05 16:37 +0900
description: ''
category: 'Machine Learning'
tags: [deep-learning-from-scratch-5, Machine Learning]
published: true
math: true
lang: en
ref: deep-learning-from-scratch-5_05
---

Since the explanation of KL divergence was somewhat brief, I've written various notes while also reviewing PRML. Whether this is directly useful for practical ML engineering work is debatable.

The explanations in Deep Learning from Scratch Volume 5 feel like they've condensed PRML's explanations for beginners in machine learning mathematics, fitting within page constraints. For my own reference, I revisited [materials I created as a student (Chapter 1)](https://speakerdeck.com/snkmr/prml-chapter-1-5-dot-0-5-dot-4?slide=22) for KL divergence, and [materials I created as a student (Chapter 9)](https://speakerdeck.com/snkmr/prml-chapter-9) for the EM algorithm.

## Information Content and Entropy

### Information Content

When a random variable takes a value $$x$$, the information content it provides is defined as follows:

$$
h(x) = -\log_2 p(x)
$$

The base of the logarithm has flexibility; when base 2 is used, the unit is called bits. For example, if $$p(x)=1/2$$, the result is 1 bit.

$$
h(x)= -\log_2(1/2)=1
$$

Consider a situation where a sender wants to convey the value of $$x$$ to a receiver. The information content obtained from a single observation is $$h(X)$$, but since $$x$$ is determined probabilistically, the average (expected value) of this is the entropy.

$$
H[X] = \mathbb{E}_{p}\left[h(X)\right]
$$

For discrete cases:

$$
H[X] = -\sum_x p(x)\log_2 p(x)
$$

### Differential Entropy

For discrete random variables, entropy can be defined straightforwardly. However, for continuous variables, $$p(x)$$ is a probability density, and the probability at any single point is 0, so we first consider discretization.

Consider quantizing a continuous probability density function into intervals of width $$\Delta$$. The probability that a data point $$x$$ falls into interval $$i$$ can be written as:

$$
p(x_i)\Delta
$$

Computing the entropy with this discretized probability:

$$
H_\Delta
= -\sum_i p(x_i)\Delta \ln\big(p(x_i)\Delta\big)
= -\sum_i p(x_i)\Delta \ln p(x_i)-\sum_i p(x_i)\Delta \ln\Delta
$$

Since $$\sum_i p(x_i)\Delta \approx \int p(x)\,dx = 1$$, the second term becomes:

$$
-\sum_i p(x_i)\Delta \ln\Delta \approx -\ln\Delta
$$

When we construct the entropy from this discretized probability and consider $$\Delta\to 0$$, the second term diverges. The first term, which remains finite in the limit $$\Delta\to 0$$, is called differential entropy $$H[x]$$:

$$
H[x] = -\int p(x)\ln p(x) dx
$$

The information required (discrete entropy) to distinguish $$x$$ "up to precision $$\Delta$$" with quantization width $$\Delta$$ is approximately:

$$
H_\Delta \approx H[x] - \ln\Delta
$$

The sender can transmit information more precisely by dividing interval $$i$$ more finely, and $$-\ln\Delta$$ represents the additional information needed as transmission precision increases. It's intuitively understandable that this diverges as $$\Delta\to 0$$. Meanwhile, $$H[x]$$ can be interpreted as the part corresponding to the spread of the distribution itself.

## KL Divergence

Consider modeling the true distribution $$p(x)$$ with an approximate distribution $$q(x)$$ for encoding. In this context, KL divergence formalizes how much extra information is needed on average when using the approximate distribution $$q(x)$$ compared to using the true distribution $$p(x)$$ for transmission.

### Introduction

When a continuous variable $$x$$ is quantized with interval width $$\Delta$$, let $$x_i$$ be the representative value of interval $$i$$. When the true distribution is $$p(x)$$, let $$P_i$$ denote the probability of interval $$i$$ as follows:

$$
P_i \approx p(x_i)\Delta
$$

The probability of interval $$i$$ computed by model $$q(x)$$ is:

$$
Q_i \approx q(x_i)\Delta
$$

Recall that the information content when a random variable takes value $$x$$ is defined as:

$$
h(x) = -\ln p(x)
$$

Therefore, when transmitting information about interval $$i$$:

* With an optimal code matched to the true distribution: $$-\ln P_i$$
* With a code created to match the approximate model $$q$$: $$-\ln Q_i$$

### KL Divergence for Discrete Distributions

Using these information contents, we compute entropy (expected value of information content).
Since data is generated from the true distribution $$p$$, information content is always weighted by $$P_i$$ in the expectation calculation.

* Using truly optimal code length with $$p(x)$$:

$$
H_\Delta(P)=\sum_i P_i(-\ln P_i)
$$

* Using a code created from approximate distribution $$q(x)$$:

$$
H_\Delta(P,Q)=\sum_i P_i(-\ln Q_i)
$$

The difference becomes "the average additional information required by using the approximate distribution $$q(x)$$":

$$
H_\Delta(P,Q)-H_\Delta(P)
= \sum_i P_i\ln\frac{P_i}{Q_i}
$$

This is the KL divergence (relative entropy), which represents the "average additional information" needed to identify the value of $$x$$ when encoding the unknown true distribution $$p(x)$$ using model $$q(x)$$.

### KL Divergence for Continuous Distributions

Since:

$$
P_i \approx p(x_i)\Delta,\quad Q_i \approx q(x_i)\Delta
$$

we have:

$$
\ln\frac{P_i}{Q_i}
= \ln\frac{p(x_i)\Delta}{q(x_i)\Delta}
= \ln\frac{p(x_i)}{q(x_i)}
$$

Thus, $$\Delta$$ can be canceled, and:

$$
\sum_i P_i\ln\frac{P_i}{Q_i}
\approx
\sum_i p(x_i)\Delta \ln\frac{p(x_i)}{q(x_i)}
$$

Taking the limit as $$\Delta \to 0$$:

$$
\sum_i p(x_i)\Delta \ln\frac{p(x_i)}{q(x_i)}
\xrightarrow[\Delta\to 0]{}
\int p(x)\ln\frac{p(x)}{q(x)} dx
$$

In differential entropy, specifying continuous values with infinite precision requires infinite bits, and $$-\ln\Delta$$ diverges. However, in KL divergence, since we take a ratio, $$\Delta$$ cancels out and the divergence disappears.

### KL Divergence Is Not a Distance

Writing KL divergence as $$\mathrm{KL}(p\|q)$$:

$$
\mathrm{KL}(p\|q)=\int p(x)\ln\frac{p(x)}{q(x)} dx
$$

In computing the expected information content, the data is weighted by the true distribution $$p$$ that generates it. In other words, where large penalties are given when distributions are far apart is determined by the true distribution $$p$$ that generates the data. Therefore, KL divergence changes its value when the order of $$p$$ and $$q$$ is swapped.

$$
\mathrm{KL}(p\|q)\neq \mathrm{KL}(q\|p)
$$

---

## EM Algorithm

In [[step_04]], it was explained that the log-likelihood of GMM takes a log-sum form, and the maximum likelihood estimation problem cannot be solved analytically. Therefore, the EM algorithm is introduced, which can iteratively solve maximum likelihood estimation and MAP estimation not only for GMM but more generally.

### Log-Sum

Let $$\mathbf{X}$$ be the observed data, $$\mathbf{Z}$$ be the latent variables, $$\boldsymbol{\theta}$$ be all parameters, and $$q(\mathbf{Z})$$ be an auxiliary distribution introduced on the latent variables.
Marginalizing over latent variables:

$$
p(\mathbf{X}|\boldsymbol{\theta})
=\sum_{\mathbf{Z}} p(\mathbf{X},\mathbf{Z}|\boldsymbol{\theta})
$$

The log marginal likelihood we want to maximize in maximum likelihood estimation is:

$$
\ln p(\mathbf{X}|\boldsymbol{\theta})
=\ln\sum_{\mathbf{Z}} p(\mathbf{X},\mathbf{Z}|\boldsymbol{\theta})
$$

Here, since $$\ln$$ is outside $$\sum_{\mathbf{Z}}$$, this cannot be solved analytically.

### ELBO

We introduce an arbitrary distribution $$q(\mathbf{Z})$$ over latent variables $$\mathbf{Z}$$.
Considering the range where $$q(\mathbf{Z})>0$$, we multiply the marginalized log-likelihood by $$\frac{q(\mathbf{Z})}{q(\mathbf{Z})}$$:

$$
\ln p(\mathbf{X}|\boldsymbol{\theta})
=\ln\sum_{\mathbf{Z}} q(\mathbf{Z})\frac{p(\mathbf{X},\mathbf{Z}|\boldsymbol{\theta})}{q(\mathbf{Z})}
$$

Applying Jensen's inequality:

$$
\ln p(\mathbf{X}|\boldsymbol{\theta})
\ge
\sum_{\mathbf{Z}} q(\mathbf{Z})
\ln\frac{p(\mathbf{X},\mathbf{Z}|\boldsymbol{\theta})}{q(\mathbf{Z})}
\equiv \mathcal{L}(q,\boldsymbol{\theta})
$$

Here, $$\mathcal{L}(q,\boldsymbol{\theta})$$ is called the Evidence Lower Bound (lower bound of the marginal likelihood = evidence).

In Deep Learning from Scratch 5, the derivation transforms the log-likelihood into a form that produces KL divergence, then derives the lower bound from the non-negativity of KL divergence. However, the proof of KL divergence's non-negativity uses Jensen's inequality, so it's the same calculation.

Also, transforming ELBO:

$$
\mathcal{L}(q,\boldsymbol{\theta})
= \sum_{\mathbf{Z}} q(\mathbf{Z})\ln p(\mathbf{X},\mathbf{Z}|\boldsymbol{\theta})
-\sum_{\mathbf{Z}} q(\mathbf{Z})\ln q(\mathbf{Z})
$$

can be decomposed this way. The second term is the entropy of $$q$$.

### Computing the Log-Likelihood

Looking at the relationship between log-likelihood and lower bound, their difference becomes KL divergence:

$$
\ln p(\mathbf{X}|\boldsymbol{\theta})-\mathcal{L}(q,\boldsymbol{\theta})=\mathrm{KL}(q\|p)
$$

The right-hand side takes the form:

$$
\mathrm{KL}(q\|p)=
-\sum_{\mathbf{Z}} q(\mathbf{Z})
\ln\frac{p(\mathbf{Z}|\mathbf{X},\boldsymbol{\theta})}{q(\mathbf{Z})}
$$

(where $$p(\mathbf{Z}|\mathbf{X},\boldsymbol{\theta})$$ is the posterior distribution of latent variables).
Using the non-negativity of KL divergence as mentioned earlier, we can understand the relationship between log-likelihood and ELBO:

$$
\ln p(\mathbf{X}|\boldsymbol{\theta}) \ge \mathcal{L}(q,\boldsymbol{\theta})
$$

The two important points are as follows, which correspond to the Expectation and Maximization steps of the EM algorithm:

* Increasing the lower bound $$\mathcal{L}$$ increases $$\ln p(\mathbf{X} \mid \boldsymbol{\theta})$$ (Expectation step)
* The difference between log-likelihood and lower bound is KL divergence; making $$q$$ match the posterior distribution makes the difference 0, and the lower bound touches the log-likelihood (Maximization step)

---

### E-Step: Optimize $$q(\mathbf{Z})$$ with fixed $$\boldsymbol{\theta}^{old}$$

Let the parameters at a given time step be $$\boldsymbol{\theta}^{old}$$. In the E-step, we maximize $$\mathcal{L}(q,\boldsymbol{\theta}^{old})$$ with respect to $$q$$ while keeping $$\boldsymbol{\theta}^{old}$$ fixed.

From the relationship between log-likelihood, lower bound, and KL divergence mentioned earlier, since log-likelihood is a constant independent of $$q$$, to maximize $$\mathcal{L}$$ with respect to $$q$$, we need to minimize KL divergence:

$$
\ln p(\mathbf{X}|\boldsymbol{\theta}^{old})
=
\mathcal{L}(q,\boldsymbol{\theta}^{old})
+
\mathrm{KL}\left(q(\mathbf{Z}) \| p(\mathbf{Z}|\mathbf{X},\boldsymbol{\theta}^{old})\right)
$$

Since KL divergence takes its minimum value of 0 when $$q(\mathbf{Z})$$ matches $$p(\mathbf{Z}\mid \mathbf{X},\boldsymbol{\theta}^{old})$$, the solution is:

$$
q^{new}(\mathbf{Z})
=
p(\mathbf{Z}|\mathbf{X},\boldsymbol{\theta}^{old})
$$

$$
\mathcal{L}(q^{new},\boldsymbol{\theta}^{old})
=
\ln p(\mathbf{X}|\boldsymbol{\theta}^{old})
$$

### M-Step: Update $$\boldsymbol{\theta}$$ with fixed $$q^{new}(\mathbf{Z})$$

In the M-step, the goal is to maximize the lower bound with respect to $$\theta$$ while keeping $$q^{new}$$ computed in the E-step fixed, to obtain $$\boldsymbol{\theta}^{new}$$:

$$
\boldsymbol{\theta}^{new}
=
\arg\max_{\boldsymbol{\theta}} \mathcal{L}(q^{new},\boldsymbol{\theta})
$$

In the following $$\mathcal{L}$$, the second term does not depend on $$\boldsymbol{\theta}$$:

$$
\mathcal{L}(q,\boldsymbol{\theta})
=
\sum_{\mathbf{Z}} q(\mathbf{Z})\ln p(\mathbf{X},\mathbf{Z}|\boldsymbol{\theta})
-
\sum_{\mathbf{Z}} q(\mathbf{Z})\ln q(\mathbf{Z})
$$

Therefore, the M-step is a maximization problem of the expected value of the complete data log-likelihood with respect to $$\mathbf{Z}$$:

$$
\sum_{\mathbf{Z}} q^{new}(\mathbf{Z})\ln p(\mathbf{X},\mathbf{Z}|\boldsymbol{\theta})
$$

Following PRML, we define this expected value as $$\mathcal{Q}(\boldsymbol{\theta},\boldsymbol{\theta}^{old})$$:

$$
\mathcal{Q}(\boldsymbol{\theta},\boldsymbol{\theta}^{old})
=
\mathbb{E}_{\mathbf{Z}|\mathbf{X},\boldsymbol{\theta}^{old}}
\left[\ln p(\mathbf{X},\mathbf{Z}|\boldsymbol{\theta})\right]
=
\sum_{\mathbf{Z}} p(\mathbf{Z}|\mathbf{X},\boldsymbol{\theta}^{old})\ln p(\mathbf{X},\mathbf{Z}|\boldsymbol{\theta})
$$

By solving this equation with $$q^{new}(\mathbf{Z})$$ obtained in the E-step, we find $$\boldsymbol{\theta}^{new}$$ that maximizes this expected value. Since $$\mathcal{Q}(\boldsymbol{\theta},\boldsymbol{\theta}^{old})$$ takes a sum-log form rather than log-sum, we can analytically derive the update equations:

$$
\boldsymbol{\theta}^{new}=\arg\max_{\boldsymbol{\theta}} \mathcal{Q}(\boldsymbol{\theta},\boldsymbol{\theta}^{old})
$$

For GMM, update equations can be obtained by differentiating with respect to the mean, covariance matrix, and mixing coefficients. Parameters are determined by repeating E-step and M-step until the change in likelihood function falls below a threshold. It can also be shown that the likelihood function always increases with the EM algorithm. I'll omit the detailed explanation here (explanations are available in Deep Learning from Scratch and PRML). In the SpeakerDeck materials linked at the top, I've posted handwritten intermediate calculations for GMM parameter computation, but I definitely couldn't do those calculations now...
