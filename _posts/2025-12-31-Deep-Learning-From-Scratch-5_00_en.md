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

I’ve been reading this recently.
[O’Reilly Japan link](https://www.oreilly.co.jp/books/9784814400591/)

I realized I’d forgotten quite a lot of what I once covered in PRML, so I wrote down various notes as a review.
These are mostly for myself, but I’ll leave them here in a blog-like format, organized by chapter.

---

# Step 1

* Probability distributions

  * For a **discrete** random variable, the y-axis directly represents probabilities.
  * For a **continuous** random variable, we instead have a **probability density function** (PDF) (p(x)).
  * The value (p(x)) is **not** a probability but a **density**, so calling it a “probability” is incorrect.
* Notation for a probability density function

  * When writing (p(a, b; c, d, e)), the variables (a, b) to the left of the semicolon are random variables, while (c, d, e) to the right represent constants such as parameters.
* Central Limit Theorem (CLT)

  * Suppose we sample (N) times from some density (p(x)) and obtain a sample set (\mathcal{G}={x^{(1)}, x^{(2)}, \dots, x^{(N)}}). Then the sample mean
    $$
    \bar{x} = \frac{x^{(1)} + x^{(2)} + \dots + x^{(N)}}{N}
    $$
    follows a normal distribution. This is the Central Limit Theorem. Since the sample mean is normally distributed, the (pre-averaging) sample sum is also normally distributed. The original distribution (p(x)) used for sampling can be *any* probability density function; it is known that the sample sum and sample mean become normal regardless of the underlying distribution, although the proof is apparently nontrivial and is omitted.

---

# Step 2

This step mainly explains maximum likelihood estimation.

## Preliminary

Given (N) data points, let the dataset be
$$
\mathcal{D}={x^{(1)},x^{(2)},\dots,x^{(N)}}.
$$
Estimating model parameters so that the model explains (\mathcal{D}) well is called **maximum likelihood estimation (MLE)**.

As an example, the book uses height data for 18-year-olds in Taiwan in 1993, which looks roughly Gaussian, and explains the process of fitting a normal distribution by MLE. We assume the observed data are generated i.i.d. from a normal distribution (p(x;\mu,\sigma)), and we determine the parameters ((\mu,\sigma)).

* i.i.d.

  * independent and identically distributed
  * Each data point is unaffected by other data points (**independent**) and is generated from the same probability distribution (**identically distributed**).

## Maximum likelihood estimation

We write the likelihood function as (p(\mathcal{D};\mu,\sigma)). If the data are i.i.d., the likelihood is the product of the density values for each data point:
$$
p(\mathcal{D};\mu,\sigma)=\prod_{n=1}^{N} p\left(x^{(n)};\mu,\sigma\right).
$$
In MLE, we choose parameters that maximize this likelihood so that (p(x)) becomes a model that explains (\mathcal{D}) well. In practice, we take logs to make computation easier and work with the log-likelihood:
$$
\log p(\mathcal{D};\mu,\sigma)
=\sum_{n=1}^{N}\log p\left(x^{(n)};\mu,\sigma\right).
$$

$$
(\hat{\mu},\hat{\sigma})
=\arg\max_{\mu,\sigma}\ \log p(\mathcal{D};\mu,\sigma).
$$

Here (p(x)) is assumed to be a simple normal distribution, so we can obtain an analytic solution for (\mu) and (\sigma). The likelihood is
$$
p(\mathcal{D};\mu,\sigma)
=\prod_{n=1}^{N}\frac{1}{\sqrt{2\pi}\sigma}
\exp \left(-\frac{(x^{(n)}-\mu)^2}{2\sigma^2}\right).
$$

Taking partial derivatives with respect to (\mu) and (\sigma) shows (details omitted) that (\hat{\mu}) is the sample mean and (\hat{\sigma}) corresponds to the variance with denominator (N):
$$
\hat{\mu}=\frac{1}{N}\sum_{n=1}^{N}x^{(n)}
$$
$$
\hat{\sigma}^2=\frac{1}{N}\sum_{n=1}^{N}\left(x^{(n)}-\hat{\mu}\right)^2.
$$

---

# Step 3

This chapter gives a brief explanation of the multivariate Gaussian distribution. There’s a figure showing how the shape changes when you vary the covariance matrix, and I looked up a few things to understand it while connecting it to the equations.

## Multivariate Gaussian distribution

When a (d)-dimensional random vector (x \in \mathbb{R}^d) follows a multivariate Gaussian distribution with mean vector (\mu \in \mathbb{R}^d) and covariance matrix (\Sigma \in \mathbb{R}^{d\times d}), we write
$$
x \sim \mathcal{N}(\mu,\Sigma)
$$
and the probability density function is given by
$$
p(x)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}
\exp\left(-\frac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)\right).
$$

### Mahalanobis distance

In the one-dimensional Gaussian, the exponent is (-((x-\mu)^2)/(2\sigma^2)). This divides by the variance, reflecting the fact that when the variance is larger, deviations from the mean are less “rare” and the density remains higher:
$$
p(x)\propto \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).
$$

In the multivariate case, the exponent is the negative half of the squared **Mahalanobis distance** (d_M(x,\mu)). The Mahalanobis distance measures distance while taking into account variance along each axis as well as correlations between axes; directions with large variance/covariance make large deviations less unusual, so the distance is effectively scaled down in those directions:
$$
d_M(x,\mu)=\sqrt{(x-\mu)^\top \Sigma^{-1}(x-\mu)}.
$$

Since the covariance matrix is symmetric ((\Sigma=\Sigma^\top)), if it is positive definite then it can be eigendecomposed:
$$
\Sigma = Q\Lambda Q^\top
$$

* (Q=[q_1,\dots,q_d]) is an orthogonal matrix ((Q^\top Q = I))
* (\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_d)) contains eigenvalues ((\lambda_i>0))
* Each (\lambda_i) corresponds to the variance in the direction of eigenvector (q_i)

Using (Q^{-1}=Q^\top), the inverse can be written as
$$
\Sigma^{-1}=(Q\Lambda Q^\top)^{-1}
=Q\Lambda^{-1}Q^\top.
$$

Substituting this into the quadratic form,
$$
(x-\mu)^\top \Sigma^{-1}(x-\mu)
= (x-\mu)^\top Q\Lambda^{-1}Q^\top (x-\mu).
$$

Here, (Q^\top (x-\mu)) is the rotated coordinate (y):
$$
y := Q^\top (x-\mu).
$$

* Why does (Q^\top (x-\mu)) represent a rotation?

  * (Q) consists of eigenvectors from the eigendecomposition; let the (i)-th eigenvector be (q_i).
  * If we express an arbitrary vector (v) in the new basis, using coordinates (y_i) along direction (q_i), we can write (with (q_i) a vector and (y_i) a scalar)
    $$ v = \sum_{i=1}^d y_i q_i $$
  * The coordinate (y_k) along the (k)-th axis can be obtained by multiplying both sides on the left by (q_k^\top). Because the (q_i) form an orthonormal basis, inner products with different axes are zero:
    $$ q_k^\top v = q_k^\top \sum_{i=1}^d y_i q_i
    = \sum_{i=1}^d y_i (q_k^\top q_i)
    = \sum_{i=1}^d y_i \delta_{ki}
    = y_k $$
  * Doing this for all axes, the updated coordinates can be written using the original vector:
    $$y=(q_1^\top, q_2^\top, \dots, q_d^\top)v=Q^{\top}v$$
  * Moreover, multiplying by an orthogonal matrix (Q) preserves norms, so distances remain unchanged:
    $$|Q^\top v|^2 = v^\top QQ^\top v = v^\top v = |v|^2 \qquad (Q^{-1}=Q^\top)$$
  * Therefore, multiplying by (Q^\top) corresponds to a rotation.
  * Setting (v=(x-\mu)) gives the expression above.

Rewriting the quadratic form using the rotated coordinates (y),
$$
(x-\mu)^\top Q\Lambda^{-1}Q^\top (x-\mu)
= y^\top \Lambda^{-1} y.
$$

Since (\Lambda^{-1}=\mathrm{diag}(1/\lambda_1,\dots,1/\lambda_d)), expanding into scalars gives
$$
y^\top \Lambda^{-1} y
= \sum_{i=1}^d \frac{y_i^2}{\lambda_i}.
$$

Thus,
$$
(x-\mu)^\top \Sigma^{-1}(x-\mu)
= \sum_{i=1}^d \frac{y_i^2}{\lambda_i}.
$$

Therefore:

* The deviation (y_i) in the direction (q_i) (i.e., along rotated coordinate axis (y_i)) is penalized by division by (\lambda_i), the variance in that direction.
* So even for the same deviation (y_i), directions with larger (\lambda_i) (larger variance) tend to have smaller (\frac{y_i^2}{\lambda_i}) → smaller Mahalanobis distance → exponent closer to 0 → higher probability density.

This expresses the idea that the density spreads out more in directions with larger covariance.

---

# Step 4

## Gaussian Mixture Model (GMM)

A model where observed data are assumed to be generated probabilistically from one of multiple Gaussian distributions. Chapter 4 explains up to the point where, unlike a single Gaussian, it becomes difficult to solve GMMs analytically.

## GMM

The multivariate Gaussian formula from [[step_03]] is:
$$
\mathcal{N}(x \mid \mu, \Sigma)
= \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}
\exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right).
$$

Now introduce:

* A discrete latent variable (class) (z \in {1,\dots,K})
* Mixing coefficients (parameters of a categorical distribution) (\phi_1,\dots,\phi_K) ((\phi_k \ge 0,\ \sum_{k=1}^K \phi_k = 1))
* Component means (\mu_k \in \mathbb{R}^D)
* Component covariances (\Sigma_k \in \mathbb{R}^{D\times D}) (symmetric positive definite)

We break the data generation process into: (i) choosing a latent component, (ii) generating a data point (x) from the Gaussian of that chosen component.

1. Choose a component:
   $$
   p(z=k) = \phi_k
   $$
2. Generate (x) from the Gaussian corresponding to the selected component:
   $$
   p(x \mid z=k) = \mathcal{N}(x \mid \mu_k, \Sigma_k)
   $$

Since the latent variable (z) is typically unobserved, we marginalize it out:
$$
p(x) = \sum_{k=1}^K p(z=k),p(x \mid z=k)
= \sum_{k=1}^K \phi_k,\mathcal{N}(x \mid \mu_k, \Sigma_k).
$$

In a GMM, (p(x)) is not a single Gaussian, but a linear combination of Gaussians, allowing it to represent a multimodal distribution.

## Likelihood for an observed dataset

Collect the parameters as
$$
\mathbf{\theta} = {\mathbf{\phi}, \mathbf{\mu}, \mathbf{\Sigma}}.
$$

For i.i.d. observations (\mathcal{D}={x^{(1)},x^{(2)},\dots,x^{(N)}}), the likelihood is
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

In the log-likelihood we want to maximize, the sum over Gaussian components remains inside the log (a log-sum structure), which makes it difficult to obtain an analytic solution. When (K=1) (a single Gaussian), the expression inside the log becomes a simple exponential form, so an analytic solution is possible. Also, if we assume the latent variable (z^{(n)}\in{1,\dots,K}) is observed, the log-sum disappears and we can derive an analytic solution:
$$
p(\mathcal{D}, z \mid \mathbf{\theta})
= \prod_{n=1}^N \phi_{z^{(n)}},\mathcal{N}(x^{(n)}\mid \mu_{z^{(n)}},\Sigma_{z^{(n)}})
$$
$$
\log p(\mathcal{D}, z \mid \mathbf{\theta})
= \sum_{n=1}^N \left[
\log \phi_{z^{(n)}} + \log \mathcal{N}(x^{(n)}\mid \mu_{z^{(n)}},\Sigma_{z^{(n)}})
\right].
$$

However, in reality (z^{(n)}) is unknown, so marginalization leads back to the log-sum form.

---

That’s all for today.
