---
layout: post
title: ゼロから作るDeep Learning 5 のメモ (ステップ1-4)
date: 2026-01-01 18:14 +0900
description: ''
category: 'Machine Learning'
tags: [deep-learning-from-scratch-5, Machine Learning]
published: true
math: true
lang: ja
ref: Deep-Learning-From-Scratch-5_00
---

# ゼロから作るDeep Learning 5巻

最近読んでいます。\
[オライリーのリンク](https://www.oreilly.co.jp/books/9784814400591/)

PRMLで一通りやった事も結構忘れていたので、復習も兼ねて色々メモりました。
ほぼ自分用ですが、一応章ごとにブログ的に書き残します。\
なんか数式がうまく表示されないことがあります。再読み込みしてみてください。

---

# Step 1

- 確率分布
	- 離散な確率変数の分布では、縦軸がそのまま確率になる
	- 一方で、連続な場合は確率密度関数 $p(x)$ の分布になる
	- $p(x)$ の値は確率ではなく確率密度なので、これを確率と呼ぶのは間違い
- 確率密度関数の表記
	- $p(a, b; c, d, e)$ と書いた場合、セミコロンの左側の$a, b$ は確率変数であり、セミコロンの右側の$c, d. e$ はパラメータなどの定数を表す。
- 中心極限定理 (Central Limit Theorem)
	- ある確率密度関数 $p(x)$ から $N$ 回サンプリングを実行することで、サンプル群 $\mathcal{G}=\{x^{(1)}, x^{(2)}, ... x^{(N)}\}$ を取得したときに、これらのサンプル平均 $$\bar{x} = \frac{x^{(1)} + x^{(2)} + ... + x^{(N)}}{N}$$
	  が正規分布になる。これを中心極限定理と呼ぶ。サンプル平均が正規分布に従うため、平均する前のサンプル和も正規分布になる。サンプリングに使う元の分布 $p(x)$ は任意の確率密度関数であり、どんな分布を使ってもサンプル和やサンプル平均が正規分布になることが知られているが、これを証明するのは難しいらしく割愛されている。

---

# Step 2

主に最尤推定の説明

## Preliminary

データ点が N 個あるとして、データの集合
$$
\mathcal{D}=\{x^{(1)},x^{(2)},\dots,x^{(N)}\}  
$$
をよく説明できるようにモデルのパラメータを推定することを最尤推定と呼ぶ。

正規分布っぽい形になっている台湾における1993年の18歳の身長のデータを使い、これにfitするように正規分布を最尤推定するプロセスを例に説明している。実データが、正規分布である $p(x;\mu,\sigma)$ から i.i.d. に生成されたものとしてみなし、パラメータ ($\mu, \sigma$) を決定する。

- i.i.d.
	- independent and identically distributed
	- それぞれのデータ点が、他のデータ点に影響されず (independent), 同じ確率分布から生成されたもの (identically distributed) であるということ


## Maximum likelihood estimation 

尤度関数を $p(D;\mu,\sigma)$ と書く。i.i.d. であれば、尤度はデータ点ごとの確率密度の積となる。

$$  
p(\mathcal{D};\mu,\sigma)=\prod_{n=1}^{N} p\left(x^{(n)};\mu,\sigma\right)  
$$
最尤推定では、この尤度関数を最大化するパラメータを決定することで、$p(x)$ が $\mathcal{D}$ をよく説明することができるモデルになるようにパラメータを推定する。実際には計算がしやすいようにlogをとり、対数尤度関数として扱う。
$$  
\log p(\mathcal{D};\mu,\sigma)  
=\sum_{n=1}^{N}\log p\left(x^{(n)};\mu,\sigma\right)  
$$

$$  
(\hat{\mu},\hat{\sigma})  
=\arg\max_{\mu,\sigma}; \log p(\mathcal{D};\mu,\sigma)
$$
今回は $p(x)$ が単純な正規分布であるという設定で最尤推定を考えているため、$\mu, \sigma$ に関して解析的な解を得ることができる。対数尤度関数

$$  
p(D;\mu,\sigma)  
=\prod_{n=1}^{N}\frac{1}{\sqrt{2\pi}\sigma}  
\exp \left(-\frac{(x^{(n)}-\mu)^2}{2\sigma^2}\right)  
$$

 を、$\mu, \sigma$ のそれぞれで偏微分すれば、$\hat{\mu}$ は標本平均になり、$\hat{\sigma}$ は分母が $N$ の分散になる (計算略)。
$$    
\hat{\mu}=\frac{1}{N}\sum_{n=1}^{N}x^{(n)}  
$$
$$  
\hat{\sigma}^2=\frac{1}{N}\sum_{n=1}^{N}\left(x^{(n)}-\hat{\mu}\right)^2  
$$


---

# Step 3

多次元ガウス分布の簡単な説明の章
分散共分散行列を変えると分布の形が変わりますという図解がありますが、数式と関連付けながら理解するためにちょっと調べものをしました。

## 多次元ガウス分布

$d$ 次元ベクトル確率変数 $x \in \mathbb{R}^d$ が 多次元ガウス分布に従うとき、平均ベクトル $\mu \in \mathbb{R}^d$、共分散行列 $\Sigma \in \mathbb{R}^{d\times d}$ として、
$$  
x \sim \mathcal{N}(\mu,\Sigma)  
$$

と書け、確率密度関数が

$$  
p(x)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}  
\exp\left(-\frac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)\right)  
$$

で与えられる。


### マハラノビス距離

1次元ガウス分布では、  指数関数の中身が $-((x-\mu)^2)/2\sigma^2$ であった。この式では分散で割っており、分散が大きいほど平均からの数値のズレが珍しくなく、確率密度が高くなることを反映している。
$$  
p(x)\propto \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right).  
$$  

指数関数の中身はマハラノビス距離 $d_M(x,\mu)$ の2乗になっている。マハラノビス距離は、各軸の分散スケールとの相関を考慮して計算した距離であり、分散、共分散が大きい方向の軸は数値が大きくずれることがめずらしくないため、距離が小さくスケーリングされる。

$$  
d_M(x,\mu)=\sqrt{(x-\mu)^\top \Sigma^{-1}(x-\mu)}.  
$$

共分散行列は対称なので（$\Sigma=\Sigma^\top$）、正定値だとすると固有分解できる。
$$  
\Sigma = Q\Lambda Q^\top  
$$

- $Q=[q_1,\dots,q_d]$ は直交行列（$Q^\top Q = I$）
- $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_d)$ は固有値（$\lambda_i>0$）
- 各 $\lambda_i$ は「固有ベクトル $q_i$ 方向の分散」に対応する

このとき逆行列は、$Q^{-1}=Q^\top$ を使って以下のように書くことができる。

$$  
\Sigma^{-1}=(Q\Lambda Q^\top)^{-1}  
=Q\Lambda^{-1}Q^\top  
$$

二次形式に固有分解を代入し、

$$  
(x-\mu)^\top \Sigma^{-1}(x-\mu)  
= (x-\mu)^\top Q\Lambda^{-1}Q^\top (x-\mu)  
$$

ここで 、 $Q^\top (x-\mu)$ は、$(x-\mu)$ を回転した座標 $y$ になっている。
$$  
y := Q^\top (x-\mu)  
$$
- なぜ$Q^\top (x-\mu)$が回転を表しているか
	- $Q$ は固有値分解して得られた固有ベクトルであり、$i$ 番目の固有ベクトルを $q_i$ とする。
	- 任意のベクトル $v$ を新しい基底ベクトルでの座標系に更新する場合について考えると、 $q_i$ 方向についての座標 $y_i$ を使って以下のように書くことができる($q_i$はベクトル、$y_i$ はスカラー)。
	  $$ v = \sum_{i=1}^d y_i q_i$$
	-  $k$ 番目の軸の更新後の座標 $y_k$ は、両辺に左から $k$ 番目の固有ベクトル $q_k^\top$ をかけることで得られる。 $q_k$ は直交基底なので、異なる軸との内積は0になる
	  $$ q_k^\top v = q_k^\top \sum_{i=1}^d y_i q_i
	  = \sum_{i=1}^d y_i (q_k^\top q_i)
	  = \sum_{i=1}^d y_i \delta_{ki}  
	  = y_k $$
	- 全ての軸に対してこれを計算すれば、更新後の座標はもとのベクトルを使って、
	  $$y=(q_1^\top, q_2^\top, ... q_N^\top)v=Q^{\top}v$$
	- さらに、直交行列 $Q$ は、任意のベクトル $v$ にかけてもノルムが変わらないため、変換しても距離が一定に保たれる。$$|Q^\top v|^2 = v^\top QQ^\top v = v^\top v = |v|^2$$$$(Q^{-1}=Q^\top)$$
	- 以上のことから、$Q^\top$をかけるのは回転であると言える。
	- $v=(x-\mu)$ とすれば、元々の式になる。


回転後の座標 $y$ を使って書き直すと、
$$  
(x-\mu)^\top Q\Lambda^{-1}Q^\top (x-\mu)  
= y^\top \Lambda^{-1} y  
$$

さらに $\Lambda^{-1}=\mathrm{diag}(1/\lambda_1,\dots,1/\lambda_d)$ なので、スカラーに展開すると
$$  
y^\top \Lambda^{-1} y  
= \sum_{i=1}^d \frac{y_i^2}{\lambda_i}  
$$

以上のことから、

$$  
\Sigma^{-1}=(Q\Lambda Q^\top)^{-1}  
= \sum_{i=1}^d \frac{y_i^2}{\lambda_i}  
$$

したがって、
- $q_i$ 方向（= 回転座標 $y_i$ の方向）のズレ $y_i$ は $\lambda_i$（その方向の分散）で割られてペナルティになる。
- つまり同じズレ $y_i$ でも $\lambda_i$ が大きい（分散が大きい）方向ほど $\frac{y_i^2}{\lambda_i}$ は小さくなりやすい = マハラノビス距離が小さい = exp(･)の中身が0に近づく = 確率密度が大きくなる

これによって、共分散が大きい方向に確率密度分布が広がるという事が表現されている。


---

# Step 4

## 混合ガウスモデル（Gaussian Mixture Model; GMM）

観測データが、複数のガウス分布のどれか 1 つから確率的に生成されるとしてモデル化したもの。４章は、単純なガウス分布とは異なり、GMMを解析的に解くことが難しいところまでの説明。

## GMM

[[step_03]] で扱った多変量ガウス分布の数式は以下
$$  
\mathcal{N}(x \mid \mu, \Sigma)  
= \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}  
\exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1}(x-\mu)\right)  
$$

ここに対して、以下を導入
- 潜在変数（クラス）を表す離散変数 $z \in {1,\dots,K}$
- 混合率（カテゴリ分布のパラメータ）$\phi_1,\dots,\phi_K$（$\phi_k \ge 0,\ \sum_{k=1}^K \phi_k = 1$）
- 各成分の平均 $\mu_k \in \mathbb{R}^D$
- 各成分の共分散 $\Sigma_k \in \mathbb{R}^{D\times D}$（対称正定値）

データが生成される過程を、(i) 潜在変数の選択、(ii) 選択した成分のガウス分布からデータ点 $x$ を生成 というステップで分割する
1. 成分の選択 $$  
    p(z=k) = \phi_k  
    $$
2. 選択された潜在変数に対応するガウス分布から $x$ を生成 $$  
    p(x \mid z=k) = \mathcal{N}(x \mid \mu_k, \Sigma_k)  
    $$

潜在変数 $z$ は通常観測しないため、$z$ に対して周辺化する。
$$  
p(x) = \sum_{k=1}^K p(z=k),p(x \mid z=k)  
= \sum_{k=1}^K \phi_k,\mathcal{N}(x \mid \mu_k, \Sigma_k)  
$$

GMMでは、$p(x)$ がガウス分布そのものではなく、ガウス分布の線形結合になっており、多峰性のガウス分布を表現することができる。


## 観測データ集合に対する尤度

パラメータをまとめて、以下のように書く。$$  
\mathbf{\theta} = {\mathbf{\phi}, \mathbf{\mu}, \mathbf{\Sigma}}
$$
独立同分布（i.i.d.）のデータ $\mathcal{D}=\{x^{(1)},x^{(2)},\dots,x^{(N)}\}$ を観測した場合についての尤度は、
$$  
p(\mathcal{D}\mid \mathbf{\theta})  
= \prod_{n=1}^N p(x^{(n)}\mid \mathbf{\theta})  
= \prod_{n=1}^N \sum_{k=1}^K \phi_k,\mathcal{N}(x^{(n)}\mid \mu_k,\Sigma_k)  
$$
対数尤度は、
$$  
\log p(\mathcal{D}\mid \mathbf{\theta})  
= \sum_{n=1}^N \log\left(\sum_{k=1}^K \phi_k,\mathcal{N}(x^{(n)}\mid \mu_k,\Sigma_k)\right)  
$$


最大化したい対数尤度の式で、log の内側にガウス分布成分の和（log-sum の構造）が残り、解析的に解くことが難しくなる。$K=1$ で単一のガウス分布である場合、logの中身が単純な指数関数になるため、解析的に解くことが可能。また、潜在変数 $z^{(n)}\in{1,\dots,K}$ が観測できると仮定すると、log-sumの形が無くなり解析解を得ることができる。
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

しかし、実際は $z^{(n)}$ が未知であるため、周辺化によって log-sum の形になる。


---

本日は一旦ここまで