# Pytorch による Score-Based Model (SBM) のシンプルな実装

CIFAR-10 を用いた Score-Based Model の実装および生成， FID による評価コードです．

![sample](created_image/seed=2026_K=5_alpha=1e-05_num=100_euler=True.png)

## SBM の概要

スコアベースモデル（ SBM ）は拡散モデルの一種です．
<br>

生成したいデータが何らかの確率分布 $p(\mathbf{x})$ に従っていると仮定します．
SBM はその確率分布の対数勾配（これをスコアと呼ぶ）
$\nabla_{\mathbf{x}} \log p(\mathbf{x})$
を学習し，学習したスコアを用いたランジュバン・モンテカルロ法による探索でそれらしいデータを生成します．

---

多くの場合，データが従う分布 $p(\mathbf{x})$ は未知なことからスコアの計算が困難です．
代わりに， SBM は解析的に求められる，元データ $\mathbf{x}$ へガウス分布に従うノイズ
$\mathbf{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$
を付加した攪乱データ

$$
\tilde{\mathbf{x}} = \mathbf{x} + \mathbf{\epsilon}
$$

を用いて， $\tilde{\mathbf{x}}$ が従う平均 $\mathbf{x}$，分散 $\sigma^{2}\mathbf{I}$ のガウス分布

$$
p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) = \frac{1}{(2\pi)^{D/2}\sigma^{D}} 
\exp\left(-\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|^2}{2\sigma^2}\right)
$$

のスコア

$$
\nabla_{\tilde{\mathbf{x}}} \log p_{\sigma}(\tilde{\mathbf{x}}|\mathbf{x}) 
= - \frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2}
= - \frac{\mathbf{\epsilon}}{\sigma^2}
$$

を学習目標とし，
ニューラルネットワーク等で表したパラメータ $\theta$ を持つスコア関数 $s_{\theta}(\tilde{\mathbf{x}}, \sigma)$ を学習します．
本実装ではスコア関数として U-Net を使用します．

---

生成したいデータは高次元なことがほとんど（例えば数字画像の MNIST なら $28 \times 28 = 784$ 次元）なため，
多様体仮説より，データが従う分布のほとんどがノイズのような状態（低密度領域）であり，
ごく一部分の領域にのみ望んだデータが集中して（高密度領域）存在していると考えられています． 
そのため，高密度領域のスコアは学習されづらく，探索の初期値は低密度領域である可能性が非常に高いことから効率的なサンプリングが困難です．
そのうえ仮に高密度領域に到達できたとしても，そこから別の高密度領域へ遷移に非常に多くのステップが要求されやすいことからモード崩壊が起こりやすい特徴もあります．

そこで SBM は次の目的関数を用いて，複数のノイズスケール $\sigma_t$ におけるスコアを学習します．  
ノイズスケールが大きい場合，多峰性分布は単峰性分布とみなせるため，様々なノイズスケール時のスコアを学習することで，
低密度から高密度領域まで幅広い領域のスコアを推定可能となります．

$$
\sum_{t=1}^{T} w_t
\mathbb{E}_{
\mathbf{x}\sim p(\mathbf{x}), \tilde{\mathbf{x}} 
\sim \mathcal{N}(\mathbf{x}, \sigma_t^{2}\mathbf{I})
}
\left[
\left\|
-\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma_{t}^2}-s_{\theta}(\tilde{\mathbf{x}}, \sigma_{t})
\right\|^2
\right]
$$

---

生成時には大きなノイズスケールから探索を開始し，徐々にノイズを小さくしながら次の更新則によるランジュバン・モンテカルロ法で遷移を繰り返します．

$$
\mathbf{x}_{t,k} = \mathbf{x}_{t,k-1} + \alpha_t s_{\theta}(\mathbf{x}_{t,k-1}, \sigma_t) + \sqrt{2\alpha_t}\,\mathbf{u}_k,
\quad \mathbf{u}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

これにより，探索初期はどの初期値からでも容易に高密度領域周辺へ遷移でき，探索が進むにつれ各データの近傍へ収束していくことから効率的なサンプリングが可能となります．
このサンプリングによって，最終的に $p(\mathbf{x})$ に従うサンプルを得ることができます．

---

## 環境

- Python 3.10
- Docker

依存関係は以下でインストールしてください．

```bash
pip install -r requirements.txt
```

## 学習

モデルの学習設定は次の通りです．
なお，以下のパラメータ設定をお勧めします．
<br>
sigma_min = 0.01，sigma_max = 50.0

```bash
python3 train.py \
--epochs 500 \
--batch_size 128 \
--lr 1e-4 \
--len_sigma 250 \
--min_sigma 0.01 \
--max_sigma 50.0 \
--save_dir ./SavedModels
```

学習済みモデルは save_dir で指定したディレクトリ（上記の例では ./SavedModels ）に保存されます．

## 生成

モデルの生成設定の一例は次の通りです．
なお，以下のパラメータ設定をお勧めします．
<br>
alpha = 1e-5，K = 5，sigma_min = 0.01，sigma_max = 50.0

```bash
python3 run.py \
--model_path ./SavedModels/UNet_epoch500.pth \
--num_images 100 \
--len_sigma 250 \
--alpha 1e-5 \
--K 5 \
--min_sigma 0.01 \
--max_sigma 50.0 \
--save_dir './created_image' \
--seed 42 \
--labels_all \
--euler \
--save-per 0.2
```

生成結果は save_dir で指定したディレクトリ（上記の例では ./created_image ）に保存されます．

## 生成結果の一例

![sample](created_image/seed=42_K=5_alpha=1e-05_num=100_euler=True.png)

## FID評価について

https://github.com/mseitzer/pytorch-fid の Frechet Inception Distance（FID）を用いてモデルの評価を行います．
評価する際の生成設定は次の通りです．

```bash
python3 fid_run.py \
--model_path ./SavedModels/UNet_epoch500.pth \
--num_images 10000 \
--batch_size 200 \
--len_sigma 250 \
--alpha 1e-5 \
--K 5 \
--min_sigma 0.01 \
--max_sigma 50.0 \
--fid_dir './fid_datas' \
--seed 2026 \
--gamma 3.0 \
--euler
```

上記の設定での評価結果は
FID:  21.25
となりました．