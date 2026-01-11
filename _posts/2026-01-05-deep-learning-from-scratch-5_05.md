---
layout: post
title: KLダイバージェンスとEMアルゴリズム (Step 5, ゼロから作るDeep Learning 5)
date: 2026-01-05 16:37 +0900
description: ''
category: ['Machine Learning', 'deep-learning-from-scratch-5']
tags: [deep-learning-from-scratch-5, Machine Learning]
published: true
math: true
lang: ja
ref: deep-learning-from-scratch-5_05
---


KLダイバージェンスの説明があまりなかったので、PRMLの復習も兼ねて色々メモ書きしました。
MLエンジニアの実務に役立つかと言われれば微妙なところです。


ゼロから作るDeep Learning 5巻の説明は、PRML の説明を機械学習の数学初心者向けに色々割愛してページ数の制限以内におさまるようにした感があります。自分で思い出す用に、KLダイバージェンスは[学生の時に作った資料（Chapter 1）](https://speakerdeck.com/snkmr/prml-chapter-1-5-dot-0-5-dot-4?slide=22)を、EMアルゴリズムは[学生の時に作った資料（Chapter 9）](https://speakerdeck.com/snkmr/prml-chapter-9)を見返しました。



## 情報量とエントロピー

### 情報量

ある確率変数が値 $$x$$ をとったとき、もたらす情報量は以下のように定義される。


$$ 
h(x) = -\log_2 p(x) 
$$


ただし、底には自由度があり、2にした場合はビットと呼ぶ。例えば (p(x)=1/2) なら1ビットとなる。


$$ 
h(x)= -\log_2(1/2)=1 
$$

送信者が$$x$$ の値を受信者に伝えたい状況を考えると、1回の観測で得られる情報量は $$h(X)$$ であるが、実際には $$x$$ は確率的に決まる状況を考えているため、平均（期待値）をとったものがエントロピーとなる。


$$ 
H[X] = \mathbb{E}_{p}\left[h(X)\right] 
$$


離散の場合は


$$ 
H[X] = -\sum_x p(x)\log_2 p(x) 
$$


### 微分エントロピー

離散な確率変数に関しては、エントロピーを簡単に定義できた。しかし、連続変数では $$p(x)$$ は確率密度であり、その点の確率密度は0であるため、一旦離散化して考える。

連続な確率密度関数を、幅 $$\Delta$$ の区間に分割して量子化した時について考える。データ点 $$x$$ が区間 $$i$$ に入る確率は、以下のように書ける。


$$ 
p(x_i)\Delta 
$$


この離散化された確率でエントロピーを計算すると


$$ 
H_\Delta 
= -\sum_i p(x_i)\Delta \ln\big(p(x_i)\Delta\big) 
= -\sum_i p(x_i)\Delta \ln p(x_i)-\sum_i p(x_i)\Delta \ln\Delta 
$$


ここで $$\sum_i p(x_i)\Delta \approx \int p(x),dx = 1$$ なので、第二項は


$$ 
-\sum_i p(x_i)\Delta \ln\Delta \approx -\ln\Delta 
$$


この離散化された確率でエントロピーを作って ($$\Delta\to 0$$) を考えると、第二項は発散する。$$\Delta\to 0$$ の極限で有限に残る第一項を微分エントロピー $$H[x]$$ と呼ぶ。


$$ 
H[x] = -\int p(x)\ln p(x) dx 
$$


量子化幅 $$\Delta$$ で “精度 $$\Delta$$ まで” $$x$$ を区別するのに必要な情報量（離散エントロピー）は、概ね以下の形になる。


$$ 
H_\Delta \approx H[x] - \ln\Delta 
$$


情報の送信者は、区間 $$i$$ を細かく区切るほど正確に情報を送信でき、$$-\ln\Delta$$ が伝達精度を上げるほど追加で必要になる情報を担っていて、これが $$\Delta\to 0$$ で無限大になるのは自然であると直感的に理解できる。一方で $$H[x]$$ は、分布そのものの広がりに対応する部分という解釈ができる。


## KLダイバージェンス

真の分布 $$p(x)$$ を、近似分布 $$q(x)$$ でモデル化して符号化するという状況を考える。ここまでの文脈で、真の分布$$p(x)$$ を情報の送信に使う場合と比較して、近似分布 $$q(x)$$ を使うことで、平均でどれだけ余分な情報が必要かを定式化したものがKLダイバージェンスである。

### 導入

連続変数 $$x$$ を区間幅 $$\Delta$$ で量子化した際、区間 $$i$$ の代表値を $$x_i$$ とすると真の分布が $$p(x)$$ のとき、区間 $$i$$ が出る確率を以下のように $$P_i$$ とする。


$$ 
P_i \approx p(x_i)\Delta 
$$


モデル $$q(x)$$ で計算される区間 $$i$$ の確率は、


$$ 
Q_i \approx q(x_i)\Delta 
$$


ある確率変数が値 $$x$$ をとったときの情報量は以下のように定義されるのであった。


$$ 
h(x) = -\ln p(x) 
$$


したがって、区間 $$i$$ の情報を送るときの情報量は、
- 真の分布に合わせた最適符号なら：$$-\ln P_i$$
- 近似モデル $$q$$ に合わせて作った符号を使うなら：$$-\ln Q_i$$
となる。


### 離散な分布におけるKLダイバージェンス

これらの情報量を使って、エントロピー（情報量の期待値）を計算する。
データは真の分布 $$p$$ から生成されるため、期待値の計算において情報量は常に $$P_i$$ で重み付けされる。

- $$p(x)$$ で真に最適な符号長を使う場合：


$$ 
H_\Delta(P)=\sum_i P_i(-\ln P_i) 
$$


- 近似分布 $$q(x)$$ で作った符号を使う場合：


$$ 
H_\Delta(P,Q)=\sum_i P_i(-\ln Q_i) 
$$

この差が、「近似分布 $$q(x)$$ を使うことで追加で必要になる情報量の平均」となる。


$$ 
H_\Delta(P,Q)-H_\Delta(P) 
= \sum_i P_i\ln\frac{P_i}{Q_i} 
$$


これが、未知の真の分布 $$p(x)$$ を $$q(x)$$ でモデル化して符号化したときに、$$x$$ の値を特定するのに必要な「追加情報量の平均」が KLダイバージェンス（相対エントロピー）である。


### 連続な分布でのKLダイバージェンス


$$ 
P_i \approx p(x_i)\Delta,\quad Q_i \approx q(x_i)\Delta 
$$

なので


$$ 
\ln\frac{P_i}{Q_i} 
= \ln\frac{p(x_i)\Delta}{q(x_i)\Delta} 
= \ln\frac{p(x_i)}{q(x_i)} 
$$


となり、$$\Delta$$ を消去することができるため、


$$ 
\sum_i P_i\ln\frac{P_i}{Q_i} 
\approx 
\sum_i p(x_i)\Delta \ln\frac{p(x_i)}{q(x_i)} 
$$


ここで、$$\Delta \xrightarrow{}{}0$$ の極限を考えると、


$$ 
\sum_i p(x_i)\Delta \ln\frac{p(x_i)}{q(x_i)} 
\xrightarrow[\Delta\to 0]{}
\int p(x)\ln\frac{p(x)}{q(x)} dx 
$$


微分エントロピーでは連続値を無限に細かく指定するには無限ビットが必要であり、 $$-\ln\Delta$$ が発散したが、KL では比を取るので $$\Delta$$ が消え、発散しなくなる。

### KLダイバージェンスは距離ではない

KLダイバージェンスを $$\mathrm{KL}(p \mid q)$$ で書くと、


$$ 
\mathrm{KL}(p \mid q)=\int p(x)\ln\frac{p(x)}{q(x)} dx
$$


情報量の期待値計算において、データが生成される真の分布 $$p$$ で重み付けられる。つまり、分布が離れているときに、どこで大きいペナルティを与えるかは、データを生成する真の分布 $$p$$ が決定する。そのためKLダイバージェンスは、$$p$$, $$q$$ の順序を入れ替えると値が変わる。


$$ 
\mathrm{KL}(p \mid q)\neq \mathrm{KL}(q \mid p) 
$$


---


## EMアルゴリズム



[[step_04]] で、GMMの対数尤度がlog-sumの形になっており、最尤推定問題が解析的に解けないことが説明されていました。そこで、GMMに限らず最尤推定やMAP推定を反復的に解くことができるEMアルゴリズムについて紹介されています。

### log-sum

観測データを $$\mathbf{X}$$、潜在変数を $$\mathbf{Z}$$、パラメータ全体を $$\boldsymbol{\theta}$$、潜在変数に導入する補助分布を $$q(\mathbf{Z})$$ とする。
潜在変数を周辺化すると


$$ 
p(\mathbf{X} \mid \boldsymbol{\theta}) 
=\sum_{\mathbf{Z}} p(\mathbf{X},\mathbf{Z} \mid \boldsymbol{\theta}) 
$$

なので、最尤推定で最大化したい対数周辺尤度は


$$ 
\ln p(\mathbf{X} \mid \boldsymbol{\theta}) 
=\ln\sum_{\mathbf{Z}} p(\mathbf{X},\mathbf{Z} \mid \boldsymbol{\theta}) 
$$

ここで、$$\ln$$ が $$\sum_{\mathbf{Z}}$$ の外側にあるため、解析的に解けない。


### ELBO

潜在変数 $$\mathbf{Z}$$ 上の任意の分布 $$q(\mathbf{Z})$$ を導入する。
$$q(\mathbf{Z})>0$$ の範囲で考えるとして、対数尤度を周辺化したものに $$\frac{q(\mathbf{Z})}{q(\mathbf{Z})}$$ をかけて、


$$ 
\ln p(\mathbf{X} \mid \boldsymbol{\theta}) 
=\ln\sum_{\mathbf{Z}} q(\mathbf{Z})\frac{p(\mathbf{X},\mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} 
$$


イェンセンの不等式を適用すると、


$$ 
\ln p(\mathbf{X} \mid \boldsymbol{\theta}) 
\ge 
\sum_{\mathbf{Z}} q(\mathbf{Z}) 
\ln\frac{p(\mathbf{X},\mathbf{Z} \mid \boldsymbol{\theta})}{q(\mathbf{Z})} 
\equiv \mathcal{L}(q,\boldsymbol{\theta}) 
$$


ここで $$\mathcal{L}(q,\boldsymbol{\theta})$$ を Evidence Lower Bound（周辺尤度＝evidence の下界）と呼ぶ。


ゼロから作るDeep Learning 5では、対数尤度からKLダイバージェンスの形ができるように式変形し、KLダイバージェンスが非負であることから下界を導出しているが、KLダイバージェンスが非負であることの証明にイェンセンの不等式を利用しているので、同じ計算をしている。



また、 ELBOを変形すると、


$$ 
\mathcal{L}(q,\boldsymbol{\theta})
= \sum_{\mathbf{Z}} q(\mathbf{Z})\ln p(\mathbf{X},\mathbf{Z} \mid \boldsymbol{\theta})
-\sum_{\mathbf{Z}} q(\mathbf{Z})\ln q(\mathbf{Z}) 
$$


のように分解できる。第二項は $$q$$ のエントロピーになっている。



### 対数尤度の計算

対数尤度と下界の関係をみると、これらの差が KL ダイバージェンスになる。


$$ 
\ln p(\mathbf{X} \mid \boldsymbol{\theta})-\mathcal{L}(q,\boldsymbol{\theta})=\mathrm{KL}(q \mid \mid p) 
$$


右辺は


$$ 
\mathrm{KL}(q \mid \mid p)=

-\sum_{\mathbf{Z}} q(\mathbf{Z}) 
\ln\frac{p(\mathbf{Z} \mid \mathbf{X},\boldsymbol{\theta})}{q(\mathbf{Z})} 
$$


という形になる（$$p(\mathbf{Z} \mid \mathbf{X},\boldsymbol{\theta})$$ は潜在変数の事後分布）。
先述の通り、KL ダイバージェンスが非負であることを利用すれば、対数尤度とELBOの関係が分かる。


$$ 
\ln p(\mathbf{X} \mid \boldsymbol{\theta}) \ge \mathcal{L}(q,\boldsymbol{\theta}) 
$$


重要なのは以下の2点で、それぞれがEMアルゴリズムとExpectationとMaximizationに対応する。

- 下界 $$\mathcal{L}$$ を大きくすることで、$$\ln p(\mathbf{X} \mid \boldsymbol{\theta})$$ が大きくなる（Expectation step）
- 対数尤度と下界の差分がは KLダイバージェンスであり、$$q$$ を事後分布に一致させれば差が 0 になって下界が対数尤度と接する（Maximization step）

---

### Eステップ：$$\boldsymbol{\theta}^{old}$$ を固定して $$q(\mathbf{Z})$$ を最適化

あるtime stepでのパラメータを $$\boldsymbol{\theta}^{old}$$ としする。Eステップでは、$$\boldsymbol{\theta}^{old}$$ を固定して $$\mathcal{L}(q,\boldsymbol{\theta}^{old})$$ を $$q$$ について最大化する。

先述の対数尤度、下界、KLダイバージェンスの関係から、対数尤度は $$q$$ に依らない定数であり、 $$\mathcal{L}$$ を $$q$$ について最大化するためには KLダイバージェンスを最小化すればよいことが分かる。


$$
\ln p(\mathbf{X} \mid \boldsymbol{\theta}^{old})
=
\mathcal{L}(q,\boldsymbol{\theta}^{old}) 
+ 
\mathrm{KL}\left(q(\mathbf{Z}) \mid \mid p(\mathbf{Z} \mid \mathbf{X},\boldsymbol{\theta}^{old})\right) 
$$


KLダイバージェンスは $$q(\mathbf{Z})$$ と、$$p(\mathbf{Z} \mid \mid \mathbf{X}, \mathbf{\theta}^{old})$$ が一致するときに最小値 0 をとるため、以下が解となる。


$$
q^{new}(\mathbf{Z})
=
p(\mathbf{Z} \mid \mathbf{X},\boldsymbol{\theta}^{old}) 
$$


$$ 
\mathcal{L}(q^{new},\boldsymbol{\theta}^{old})
=
\ln p(\mathbf{X} \mid \boldsymbol{\theta}^{old}) 
$$


### Mステップ：$$q^{new}(\mathbf{Z})$$ を固定して $$\boldsymbol{\theta}$$ を更新

Mステップでは、Eステップで計算した $$q^{new}$$ を固定して、下界を $$\theta$$ に対して最大化し、$$\mathbf{\theta}^{new}$$
を得ることを目的とする。
 

$$ 
\boldsymbol{\theta}^{new}
=
\arg\max_{\boldsymbol{\theta}} \mathcal{L}(q^{new},\boldsymbol{\theta}) 
$$


以下の $$\mathcal{L}$$ で、第二項は $$\boldsymbol{\theta}$$ に依らない。


$$ 
\mathcal{L}(q,\boldsymbol{\theta})
=
\sum_{\mathbf{Z}} q(\mathbf{Z})\ln p(\mathbf{X},\mathbf{Z} \mid \boldsymbol{\theta})
-
\sum_{\mathbf{Z}} q(\mathbf{Z})\ln q(\mathbf{Z}) 
$$


したがって、Mステップは第一項の完全データ対数尤度の、$$\mathbf{Z}$$ に関する期待値の最大化問題である。


$$ 
\sum_{\mathbf{Z}} q^{new}(\mathbf{Z})\ln p(\mathbf{X},\mathbf{Z} \mid \boldsymbol{\theta}) 
$$


この期待値を、PRMLと同様に $$\mathcal{Q}(\boldsymbol{\theta},\boldsymbol{\theta}^{old})$$ と定義する。



$$ 
\mathcal{Q}(\boldsymbol{\theta},\boldsymbol{\theta}^{old})
=
\mathbb{E}_{\mathbf{Z} \mid \mathbf{X},\boldsymbol{\theta}^{old}} 
\left[\ln p(\mathbf{X},\mathbf{Z} \mid \boldsymbol{\theta})\right]
=
\sum_{\mathbf{Z}} p(\mathbf{Z} \mid \mathbf{X},\boldsymbol{\theta}^{old})\ln p(\mathbf{X},\mathbf{Z} \mid \boldsymbol{\theta}) 
$$


Eステップで得た $$q^{new}(\mathbf{Z})$$ でこの式を解くことで、この期待値を最大化する $$\boldsymbol{\theta}^{new}$$ を探す。$$\mathcal{Q}(\boldsymbol{\theta},\boldsymbol{\theta}^{old})$$ はlog-sumではなくsum-logの形になっているので、解析的に更新式を得ることができる。 


$$ 
\boldsymbol{\theta}^{new}=\arg\max_{\boldsymbol{\theta}} \mathcal{Q}(\boldsymbol{\theta},\boldsymbol{\theta}^{old}) 
$$


GMMでは、平均、分散共分散行列、混合係数に対して微分することで更新式を得ることができる。Eステップ、Mステップを繰り返し、尤度関数の変化量が閾値を下回れば終了することでパラメータを決定する。EMアルゴリズムで、尤度関数が常に大きくなることも示すことができる。このあたりの説明は割愛（ゼロから作るDeep LearningやPRMLに解説あり）。一番上に貼ったspearkerdeckの資料でGMMのパラメータ計算で手書きの途中式とか貼っているけど今となっては間違いなく計算できない。。
