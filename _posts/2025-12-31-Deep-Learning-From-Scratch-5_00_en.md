---
layout: post
title: Notes on Deep Learning from Scratch 5 (Steps 1–4)
date: 2026-01-01 18:14 +0900
description: ''
category: 'Machine Learning'
tags: [deep-learning-from-scratch-5, Machine Learning]
published: true
math: true
lang: en
ref: Deep-Learning-From-Scratch-5_00
---


# *Deep Learning from Scratch*, Volume 5

I’ve been reading it recently.
[O’Reilly Japan link](https://www.oreilly.co.jp/books/9784814400591/)

I’d also forgotten quite a bit of what I had gone through in PRML, so I took various notes as a review.
These are mostly for myself, but I’ll leave them here as blog-style notes chapter by chapter.
Sometimes the formulas don’t render correctly. If that happens, try reloading the page.

---

# Step 1

* Probability distributions

  * For distributions of **discrete** random variables, the vertical axis directly represents probability.
  * For **continuous** variables, we deal with a probability density function (PDF) (p(x)).
  * The value of (p(x)) is not a probability but a **density**, so it is incorrect to call it a probability.
* Notation for probability density functions

  * When writing (p(a, b; c, d, e)), the (a, b) to the left of the semicolon are random variables, while (c, d, e) to the right are constants such as parameters.
* Central Limit Theorem (CLT)

  * Suppose we sample (N) times from a probability density function (p(x)) and obtain a set of samples (\mathcal{G}={x^{(1)}, x^{(2)}, \dots, x^{(N)}}). Then the sample mean
    $$
    \bar{x} = \frac{x^{(1)} + x^{(2)} + \dots + x^{(N)}}{N}
    $$
    follows a normal distribution. This is the Central Limit Theorem. Since the sample mean becomes normally distributed, the (pre-averaging) sum of samples also becomes normally distributed. The original distribution (p(x)) used for sampling can be any PDF: it is known that the sample sum/mean becomes normal regardless of the underlying distribution, though proving this is apparently difficult and is omitted in the book.

---

# Step 2

Mainly an explanation of maximum likelihood estimation.

## Preliminary

Given (N) data points, the dataset

$$
\mathcal{D}={x^{(1)},x^{(2)},\dots,x^{(N)}}
$$

Maximum likelihood estimation (MLE) refers to estimating model parameters so that the model can explain (\mathcal{D}) well.

The book explains MLE using an example: height data of 18-year-olds in Taiwan in 1993, which looks roughly normally distributed. We treat the observed data as if it were generated i.i.d. from a normal distribution (p(x;\mu,\sigma)), and determine the parameters ((\mu,\sigma)).

* i.i.d.

  * independent and identically distributed
  * each data point is not influenced by other data points (**independent**) and is generated from the same probability distribution (**identically distributed**)

## Maximum likelihood estimation

Write the likelihood function as (p(\mathcal{D};\mu,\sigma)). Under the i.i.d. assumption, the likelihood is the product of the per-datapoint densities:

$$
p(\mathcal{D};\mu,\sigma)=\prod_{n=1}^{N} p\left(x^{(n)};\mu,\sigma\right)
$$

In MLE, we choose the parameters that maximize this likelihood so that (p(x)) becomes a model that explains (\mathcal{D}) well. In practice, we take the log to make computation easier and work with the log-likelihood:

$$
\log p(\mathcal{D};\mu,\sigma)
=\sum_{n=1}^{N}\log p\left(x^{(n)};\mu,\sigma\right)
$$

$$
(\hat{\mu},\hat{\sigma})
=\arg\max_{\mu,\sigma}; \log p(\mathcal{D};\mu,\sigma)
$$

Here, since (p(x)) is assumed to be a simple normal distribution, we can obtain an analytical solution for (\mu) and (\sigma). The likelihood is

$$
p(\mathcal{D};\mu,\sigma)
=\prod_{n=1}^{N}\frac{1}{\sqrt{2\pi}\sigma}
\exp \left(-\frac{(x^{(n)}-\mu)^2}{2\sigma^2}\right)
$$

Taking partial derivatives with respect to (\mu) and (\sigma), (\hat{\mu}) becomes the sample mean, and (\hat{\sigma}) becomes the variance with (N) in the denominator (derivation omitted):

$$
\hat{\mu}=\frac{1}{N}\sum_{n=1}^{N}x^{(n)}
$$

$$
\hat{\sigma}^2=\frac{1}{N}\sum_{n=1}^{N}\left(x^{(n)}-\hat{\mu}\right)^2
$$

---

# Step 3

A chapter giving a brief explanation of the multivariate Gaussian distribution.
There is an illustration showing how changing the covariance matrix changes the shape of the distribution; I looked up a few things to understand it while connecting it to the formulas.

## Multivariate Gaussian distribution

When a (d)-dimensional vector random variable (x \in \mathbb{R}^d) follows a multivariate Gaussian distribution with mean vector (\mu \in \mathbb{R}^d) and covariance matrix (\Sigma \in \mathbb{R}^{d\times d}),

$$
x \sim \mathcal{N}(\mu,\Sigma)
$$

and its probability density function is given by

$$
p(x)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(-\frac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)\right).
$$

### Mahalanobis distance

In the one-dimensional Gaussian distribution, the exponent is (-((x-\mu)^2)/(2\sigma^2)). This divides by the variance, reflecting the fact that when the variance is large, deviations from the mean are less “rare” and the density is higher.

$$
p(x)\propto \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
$$

In the multivariate case, the exponent is the squared Mahalanobis distance (d_M(x,\mu)). The Mahalanobis distance is a distance computed while accounting for the variance scale and correlations of each axis; along directions with large variance/covariance, large deviations are not unusual, so the distance is scaled down.

$$
d_M(x,\mu)=\sqrt{(x-\mu)^\top \Sigma^{-1}(x-\mu)}.
$$

Since the covariance matrix is symmetric ((\Sigma=\Sigma^\top)), if it is positive definite, we can perform an eigen-decomposition:

$$
\Sigma = Q\Lambda Q^\top
$$

* (Q=[q_1,\dots,q_d]) is an orthogonal matrix ((Q^\top Q = I))
* (\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_d)) contains the eigenvalues ((\lambda_i>0))
* each (\lambda_i) corresponds to the variance along the eigenvector direction (q_i)

The inverse can be written using (Q^{-1}=Q^\top):

$$
\Sigma^{-1}=(Q\Lambda Q^\top)^{-1}
=Q\Lambda^{-1}Q^\top
$$

Substituting into the quadratic form:

$$
(x-\mu)^\top \Sigma^{-1}(x-\mu)
= (x-\mu)^\top Q\Lambda^{-1}Q^\top (x-\mu)
$$

Define the rotated coordinates (y) as

$$
y := Q^\top (x-\mu).
$$

* Why does (Q^\top (x-\mu)) represent a rotation?

  * (Q) is formed by the eigenvectors from the eigen-decomposition, with the (i)-th eigenvector denoted (q_i).
  * Consider expressing an arbitrary vector (v) in the coordinate system of the new basis vectors. Using coordinates (y_i) along direction (q_i), we can write (with (q_i) a vector and (y_i) a scalar):
    $$
    v = \sum_{i=1}^d y_i q_i
    $$
  * The (k)-th coordinate (y_k) in the new basis is obtained by left-multiplying by (q_k^\top). Since ({q_i}) are an orthonormal basis, inner products with different axes become 0:
    $$
    q_k^\top v = q_k^\top \sum_{i=1}^d y_i q_i
    = \sum_{i=1}^d y_i (q_k^\top q_i)
    = \sum_{i=1}^d y_i \delta_{ki}
    = y_k
    $$
  * Doing this for all axes gives
    $$
    y=(q_1^\top, q_2^\top, \dots, q_d^\top)v = Q^{\top}v.
    $$
  * Moreover, for an orthogonal matrix (Q), multiplying any vector (v) does not change its norm, so distances are preserved:
    $$
    \lVert Q^\top v\rVert^2 = v^\top QQ^\top v = v^\top v = \lVert v\rVert^2
    \qquad (Q^{-1}=Q^\top)
    $$
  * Therefore, multiplying by (Q^\top) corresponds to a rotation.
  * Setting (v=(x-\mu)) recovers the original expression.

Rewriting with (y):

$$
(x-\mu)^\top Q\Lambda^{-1}Q^\top (x-\mu)
= y^\top \Lambda^{-1} y
$$

Since (\Lambda^{-1}=\mathrm{diag}(1/\lambda_1,\dots,1/\lambda_d)), expanding to a scalar form yields

$$
y^\top \Lambda^{-1} y
= \sum_{i=1}^d \frac{y_i^2}{\lambda_i}.
$$

From this,

$$
(x-\mu)^\top \Sigma^{-1}(x-\mu)
= \sum_{i=1}^d \frac{y_i^2}{\lambda_i}.
$$

Therefore:

* The deviation (y_i) along direction (q_i) (i.e., the rotated coordinate axis) is penalized by division by (\lambda_i) (the variance in that direction).
* For the same deviation (y_i), a larger (\lambda_i) (larger variance) makes (\frac{y_i^2}{\lambda_i}) smaller more easily
  (\Rightarrow) Mahalanobis distance is smaller
  (\Rightarrow) the exponent gets closer to 0
  (\Rightarrow) probability density becomes larger.

This expresses how the density spreads out along directions with large covariance.

---

# Step 4

## Gaussian Mixture Model (GMM)

A model in which observed data are assumed to be generated probabilistically from one of multiple Gaussian distributions. Chapter 4 explains up to the point where, unlike a single Gaussian distribution, it becomes difficult to solve GMM analytically.

## GMM

The multivariate Gaussian distribution formula used in [[step_03]] is:

$$
\mathcal{N}(x \mid \mu, \Sigma)
= \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}
\exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right)
$$

Now introduce:

* a discrete variable (z \in {1,\dots,K}) representing a latent variable (class)
* mixing coefficients (parameters of a categorical distribution) (\phi_1,\dots,\phi_K)
  ((\phi_k \ge 0,\ \sum_{k=1}^K \phi_k = 1))
* mean of each component (\mu_k \in \mathbb{R}^D)
* covariance of each component (\Sigma_k \in \mathbb{R}^{D\times D}) (symmetric positive definite)

We split the data generation process into steps: (i) choose a latent variable, (ii) generate a data point (x) from the Gaussian corresponding to the chosen component.

1. Choose a component

   $$
   p(z=k) = \phi_k
   $$

2. Generate (x) from the Gaussian corresponding to the chosen latent variable

   $$
   p(x \mid z=k) = \mathcal{N}(x \mid \mu_k, \Sigma_k)
   $$

Since the latent variable (z) is usually unobserved, we marginalize it out:

$$
p(x) = \sum_{k=1}^K p(z=k),p(x \mid z=k)
= \sum_{k=1}^K \phi_k,\mathcal{N}(x \mid \mu_k, \Sigma_k).
$$

In a GMM, (p(x)) is not a Gaussian distribution itself but a linear combination of Gaussians, so it can represent a multimodal distribution.

## Likelihood for the observed dataset

Collect the parameters as

$$
\mathbf{\theta} = {\mathbf{\phi}, \mathbf{\mu}, \mathbf{\Sigma}}.
$$

For i.i.d. data (\mathcal{D}={x^{(1)},x^{(2)},\dots,x^{(N)}}), the likelihood is

$$
p(\mathcal{D}\mid \mathbf{\theta})
= \prod_{n=1}^N p(x^{(n)}\mid \mathbf{\theta})
= \prod_{n=1}^N \sum_{k=1}^K \phi_k,\mathcal{N}(x^{(n)}\mid \mu_k,\Sigma_k).
$$

The log-likelihood is

$$
\log p(\mathcal{D}\mid \mathbf{\theta})
= \sum_{n=1}^N \log\left(\sum_{k=1}^K \phi_k,\mathcal{N}(x^{(n)}\mid \mu_k,\Sigma_k)\right).
$$

In the log-likelihood we want to maximize, the sum over Gaussian components remains inside the log (a log-sum structure), making it difficult to obtain an analytical solution. If (K=1) (a single Gaussian), the inside of the log becomes a simple exponential form, so an analytical solution is possible. Also, if we assume the latent variables (z^{(n)}\in{1,\dots,K}) are observable, the log-sum form disappears and we can derive a closed-form solution.

$$
p(\mathcal{D}, z \mid \mathbf{\theta})
= \prod_{n=1}^N \phi_{z^{(n)}},\mathcal{N}(x^{(n)}\mid \mu_{z^{(n)}},\Sigma_{z^{(n)}})
$$

$$
\log p(\mathcal{D}, z \mid \mathbf{\theta})
= \sum_{n=1}^N \left[
\log \phi_{z^{(n)}} + \log \mathcal{N}(x^{(n)}\mid \mu_{z^{(n)}},\Sigma_{z^{(n)}})
\right]
$$

However, in reality (z^{(n)}) is unknown, so marginalization leads back to the log-sum form.

---

That’s all for today.
