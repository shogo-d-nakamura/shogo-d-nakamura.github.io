---
layout: post
title: Optimizers
date: 2026-01-11 17:00 +0900
description: ''
category: ['Machine Learning', 'deep-learning-from-scratch-5']
tags: [Machine Learning, deep-learning-from-scratch-5]
published: true
math: true
lang: en
ref: Optimizers_en
---

This is a continuation of the previous post.
I have summarized RMSProp, AdaGrad, AdaDelta, and Adam.


## Limitations of Momentum

As explained previously, the optimal learning rate depends on the local curvature of the error surface. Furthermore, this curvature can vary depending on the direction in the parameter space - a small learning rate is desirable in steep directions to prevent oscillation, while a larger learning rate is preferable in gentle directions to achieve faster convergence. However, standard gradient descent uses the same learning rate $$\eta$$ for all parameters, making it difficult to examine the curvature at each point in parameter space and adjust the learning rate appropriately.

Therefore, algorithms have been developed that use different learning rates for each parameter in the network and automatically adjust these values during training.

---

## AdaGrad (Adaptive Gradient)

If the gradient for a certain parameter is consistently large, this suggests that the error surface is steep in the direction of that parameter. In such cases, the learning rate should be reduced to make smaller parameter updates. Conversely, parameters with small gradients can be safely updated with larger learning rates. AdaGrad uses the cumulative sum of squared gradients computed for each parameter to decrease each learning rate parameter over time.


### Update Rule

First, for each parameter $$w_i$$, we prepare a variable $$r_i$$ that accumulates the squared gradients. $$r_i$$ is updated as follows:


$$r_i^{(\tau)} = r_i^{(\tau-1)} + \left( \frac{\partial E(\mathbf{w})}{\partial w_i} \right)^2$$

Next, we use this accumulated value to update the parameters:


$$w_i^{(\tau)} = w_i^{(\tau-1)} - \frac{\eta}{\sqrt{r_i^{\tau} + \delta}} \left( \frac{\partial E(\mathbf{w})}{\partial w_i} \right)$$

- $$\eta$$: Learning rate
- $$\delta$$: Small constant to avoid division by zero



The cumulative squared gradient is initialized with $$r_i^{(0)} = 0$$.


This update rule is similar to standard stochastic gradient descent, but each parameter has its own effective learning rate

$$\eta / \sqrt{r_i^{\tau} + \delta}$$
where larger accumulated squared gradients result in smaller update steps. This selectively reduces the effective learning rate for parameters with steep gradients.

### Problems with AdaGrad

In AdaGrad, since squared gradients are accumulated from the beginning of training, $$r_i$$ increases monotonically. As a result, the effective learning rate $$\eta / \sqrt{r_i + \delta}$$ decreases monotonically over time, approaching zero.


In the later stages of training, the learning rate may become too small, causing parameters to barely update, and training may effectively stop before reaching the optimal solution.
RMSProp and AdaDelta were proposed to solve this problem.


## RMSProp (Root Mean Square Propagation)

RMSProp solved the problem of monotonically decreasing learning rates by replacing AdaGrad's simple cumulative sum of squared gradients with an exponentially weighted average.


### Update Rule

The update rule for RMSProp is as follows:

$$r_i^{(\tau)} = \beta r_i^{(\tau-1)} + (1 - \beta) \left( \frac{\partial E(\mathbf{w})}{\partial w_i} \right)^2$$

$$w_i^{(\tau)} = w_i^{(\tau-1)} - \frac{\eta}{\sqrt{r_i^{\tau} + \delta}} \left( \frac{\partial E(\mathbf{w})}{\partial w_i} \right)$$

Here, $$\beta$$ is a coefficient that determines the mixing ratio between the previous timestep's accumulated value and the squared gradient, satisfying $$0 < \beta < 1$$. A typical value of $$\beta = 0.9$$ is commonly used.


### Advantages of Exponentially Weighted Average

By using an exponentially weighted average, old gradient information gradually fades and recent gradient information is given more weight. This prevents the excessive decrease in learning rate that is a problem with AdaGrad.

The exponentially weighted average considers all past values, but as time passes, the weight on old accumulated sums decays exponentially because $$\beta$$ is repeatedly multiplied.

### Comparison with AdaGrad

In AdaGrad, denoting the squared gradient as $$g_i^2$$:


$$r_i^{(\tau)} = r_i^{(\tau-1)} + g_i^2$$


In RMSProp:


$$
r_i^{(\tau)} = \beta r_i^{(\tau-1)} + (1-\beta) g_i^2
$$


Thus, past squared gradients decay by the coefficient $$\beta$$.


For example, with $$\beta = 0.9$$, information from 10 steps ago is decayed to about $$0.9^{10} \approx 0.35$$ times the weight of current information. This prevents $$r_i$$ from increasing infinitely and allows it to converge to a value that well reflects recent gradient information.


## AdaDelta

Another algorithm proposed to solve AdaGrad's problems is AdaDelta. Like RMSProp, it uses an exponentially weighted moving average, but a major feature of AdaDelta is that it does not explicitly set the learning rate parameter $$\eta$$.

### Unit Consistency

Consider the standard gradient descent update rule with gradient $$g$$:


$$
\Delta w = -\eta \cdot g
$$

The left side $$\Delta w$$ is the change in parameters and has the same units as the parameters. On the other hand, the gradient on the right side $$g = \partial E / \partial w$$ is the error function (assumed dimensionless) differentiated by the weights, so the units do not match. Therefore, the learning rate $$\eta$$ is needed to match the dimensional units. The authors claim that this is one reason why the learning rate needs to be set manually.


### Update Rule

AdaDelta uses not only the exponentially weighted moving average of squared gradients but also the exponentially weighted moving average of squared parameter updates.


First, like RMSProp, compute the exponentially weighted moving average of squared gradients:


$$
r_i^{(\tau)} = \rho \ r_i^{(\tau-1)} + (1 - \rho) \left( \frac{\partial E(\mathbf{w})}{\partial w_i} \right)^2
$$


Next, compute the exponentially weighted moving average of squared parameter changes:


$$
s_i^{(\tau)} = \rho \ s_i^{(\tau-1)} + (1 - \rho) \left( \Delta w_i^{(\tau-1)} \right)^2
$$


Using these, define the RMS (Root Mean Square) values:


$$
\text{RMS}[g]_i^{(\tau)} = \sqrt{r_i^{(\tau)} + \delta}
$$

$$
\text{RMS}[\Delta w]_i^{(\tau)} = \sqrt{s_i^{(\tau)} + \delta}
$$


Using these, we obtain the final parameter update rule:


$$
\Delta w_i^{(\tau)} = - \frac{\text{RMS}[\Delta w]_i^{(\tau-1)}}{\text{RMS}[g]_i^{(\tau)}} \left( \frac{\partial E(\mathbf{w})}{\partial w_i} \right)
$$

$$w_i^{(\tau)} = w_i^{(\tau-1)} + \Delta w_i^{(\tau)}$$

Here, $$\rho$$ is the decay rate, typically $$\rho = 0.95$$ is used.

### Characteristics of AdaDelta


Looking at the units, the numerator $$\text{RMS}[\Delta w]$$ has the same units as the parameters, and the denominator $$\text{RMS}[g]$$ has the same units as the gradient. From this, the product of $$\text{RMS}[g]$$ and the gradient $$\frac{\partial E(\mathbf{w})}{\partial w_i}$$ becomes dimensionless, and the units match those of the parameters.

Also, in the initial stage, updates start small because $$s_i$$ is close to zero, and as training progresses, the update amount is automatically adjusted appropriately.


From the above, we can see that in AdaDelta, the learning rate $$\eta$$ is not included in the update rule, and instead the RMS of past parameter updates ($$\text{RMS}[\Delta w]_i^{(\tau-1)}$$) plays the role of the learning rate.



## Adam (Adaptive Moments)

In 2014, Adam was proposed, combining RMSProp and Momentum.
Adam maintains separately for each parameter an exponentially weighted moving average of gradients $$s_i$$ and an exponentially weighted moving average of squared gradients $$r_i$$, using both to formulate the update rule.

### Update Rule

Estimation of the first moment (mean) of gradients:


$$
s_i^{(\tau)} = \beta_1 s_i^{(\tau-1)} + (1 - \beta_1) \left( \frac{\partial E(\mathbf{w})}{\partial w_i} \right)
$$

Estimation of the second moment (variance) of gradients:


$$
r_i^{(\tau)} = \beta_2 r_i^{(\tau-1)} + (1 - \beta_2) \left( \frac{\partial E(\mathbf{w})}{\partial w_i} \right)^2
$$

Bias correction:


$$\hat{s}_i^{(\tau)} = \frac{s_i^{(\tau)}}{1 - \beta_1^{\tau}}$$

$$\hat{r}_i^{(\tau)} = \frac{r_i^{(\tau)}}{1 - \beta_2^{\tau}}$$

Parameter update:


$$w_i^{(\tau)} = w_i^{(\tau-1)} - \eta \frac{\hat{s}_i^{(\tau)}}{\sqrt{\hat{r}_i^{(\tau)}} + \delta}$$


#### First Moment

$$s_i$$ is the exponentially weighted moving average of gradients, which accumulates past gradient information, accelerating updates when the gradient direction is consistent and dampening updates when the direction changes frequently, similar to Momentum.

#### Second Moment

$$r_i$$ is the exponentially weighted moving average of squared gradients, playing the same role as RMSProp. This allows setting an appropriate learning rate for each parameter.
In the update rule, $$\hat{s}_i$$ is scaled by $$\eta / (\sqrt{\hat{r}_i} + \delta)$$ using the second moment and learning rate.


#### Bias Correction

In Adam, the first moment (mean) and second moment (variance) of gradients are estimated using exponentially weighted moving averages. Denoting the gradient as $$g_i$$, this can be written as:


$$s_i^{(\tau)} = \beta_1 s_i^{(\tau-1)} + (1 - \beta_1) g_i^{(\tau)}$$

$$r_i^{(\tau)} = \beta_2 r_i^{(\tau-1)} + (1 - \beta_2) (g_i^{(\tau)})^2$$


When these moving averages are initialized with $$s_i^{(0)} = 0$$ and $$r_i^{(0)} = 0$$, a problem arises where the initial estimates become smaller than the true values.


- Expansion of Moving Average

Setting $$s_i^{(0)} = 0$$ and expanding the first few steps:


$$s_i^{(1)} = \beta_1 \cdot 0 + (1 - \beta_1) g_i^{(1)} = (1 - \beta_1) g_i^{(1)}$$

$$s_i^{(2)} = \beta_1 s_i^{(1)} + (1 - \beta_1) g_i^{(2)} = \beta_1 (1 - \beta_1) g_i^{(1)} + (1 - \beta_1) g_i^{(2)}$$

$$s_i^{(3)} = \beta_1^2 (1 - \beta_1) g_i^{(1)} + \beta_1 (1 - \beta_1) g_i^{(2)} + (1 - \beta_1) g_i^{(3)}$$

In general, at time $$\tau$$:


$$
s_i^{(\tau)} = (1 - \beta_1) \sum_{k=1}^{\tau} \beta_1^{\tau - k} g_i^{(k)}
$$

- Computing the Expected Value

Assume that gradients $$g_i^{(k)}$$ are independently drawn from the same distribution with expected value $$\mathbb{E}[g_i] = \mu$$.

The expected value of $$s_i^{(\tau)}$$ is:


$$
\mathbb{E}[s_i^{(\tau)}] = (1 - \beta_1) \sum_{k=1}^{\tau} \beta_1^{\tau - k} \mathbb{E}[g_i^{(k)}] = (1 - \beta_1) \mu \sum_{k=1}^{\tau} \beta_1^{\tau - k}
$$

- Sum of Geometric Series

$$\sum_{k=1}^{\tau} \beta_1^{\tau - k}$$ is the sum of a geometric series. Substituting $$j = \tau - k$$:


$$
\sum_{k=1}^{\tau} \beta_1^{\tau - k} = \sum_{j=0}^{\tau-1} \beta_1^j = \frac{1 - \beta_1^{\tau}}{1 - \beta_1}
$$

Therefore:


$$
\mathbb{E}[s_i^{(\tau)}] = (1 - \beta_1) \mu \cdot \frac{1 - \beta_1^{\tau}}{1 - \beta_1} = \mu (1 - \beta_1^{\tau})
$$

- Bias

The true expected value is $$\mu$$, but the expected value of the estimate became $$\mu (1 - \beta_1^{\tau})$$.
This means there is a bias by the factor $$(1 - \beta_1^{\tau})$$, and since $$\beta_1^{\tau}$$ is less than 1, the estimate is smaller than the true value. Correction is needed to cancel this bias.


For example, with $$\beta_1 = 0.9$$, the bias is as follows. It can be seen that especially in early training, the estimates are undervalued compared to the true values, significantly affecting the effective learning rate.

|Time $$\tau$$|$$(1 - \beta_1^{\tau})$$|Degree of Bias|
|---|---|---|
|1|$$1 - 0.9 = 0.1$$|Only 10% of true value is estimated|
|2|$$1 - 0.81 = 0.19$$|19% of true value|
|5|$$1 - 0.59 = 0.41$$|41% of true value|
|10|$$1 - 0.35 = 0.65$$|65% of true value|
|50|$$1 - 0.005 \approx 0.995$$|Almost 100%|


{: .prompt-tip }
> **Summary**
> - **Limitations of Momentum**: Standard gradient descent uses the same learning rate for all parameters, making it impossible to set optimal learning rates for directions with different curvatures.
> - **AdaGrad**: Accumulates squared gradients for each parameter, reducing the learning rate more for larger accumulated values. However, since the accumulated value monotonically increases, the learning rate becomes excessively small in later training.
> - **RMSProp**: Replaces AdaGrad's cumulative sum with an exponentially weighted average, decaying old gradient information to prevent excessive decrease in learning rate.
> - **AdaDelta**: Uses exponentially weighted averages like RMSProp, but uses the RMS of parameter updates instead of the learning rate, eliminating the need for explicit learning rate setting.
> - **Adam**: Combines RMSProp and Momentum, utilizing both the first moment (directional consistency) and second moment (adaptive learning rate) of gradients. It also has a mechanism to correct bias from zero initialization.



---

## Random Thoughts

1. While writing this with help from ChatGPT, I learned about a recent paper suggesting that Adam's bias correction terms are essentially just doing what warmup does in learning rate schedulers. They showed that by removing the bias terms and using schedulers known to work well empirically for LLMs, the loss converges similarly[^1].

2. Since I only see Adam in papers, the combination of RMSProp and Momentum has become dominant, but I wonder why the combination of AdaDelta and Momentum didn't become popular. Is it because unlike Adam, there's no clever initialization? But then bias correction is said to be the same as learning rate schedulers, and while PyTorch has an implementation[^2], even though not requiring lr setting is supposed to be its selling point, it allows setting lr as an optional parameter. I couldn't quite figure it out.

---

## Reference


[^1]: Laing, S., & Orvieto, A. (2025). Adam Simplified: Bias Correction Debunked. arXiv:2511.20516. https://arxiv.org/abs/2511.20516

[^2]: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adadelta.html accessed on 2026-01-11.