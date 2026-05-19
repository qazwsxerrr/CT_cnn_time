# 乘性噪声下 Morozov 偏差原则的噪声半径估计

## 1. 目标

Morozov 偏差原则用于选择正则化强度或约束半径，使重建结果满足

$$
\|A c_\lambda - g^\delta\|_2 \approx \delta.
$$

其中：

- $g^\delta$ 是实际观测到的含噪数据；
- $g$ 是未知的无噪声观测；
- $\delta$ 是噪声水平或噪声半径；
- $c_\lambda$ 是由正则化问题得到的系数图像。

本项目要求：

> 可以假设噪声水平已知，但不能把 $g$ 或 `g_clean` 作为算法输入。

因此，Morozov 选参时不能使用

$$
\|g^\delta - g\|_2,
$$

而应只使用 $g^\delta$ 和已知噪声水平来估计 $\delta$。

---

## 2. 当前乘性噪声模型

当前代码中的乘性噪声为

```python
rand_u = 2.0 * torch.rand_like(g_clean) - 1.0
g_observed = g_clean + noise_level * g_clean * rand_u
```

数学形式是

$$
g_i^\delta = g_i(1 + \alpha \xi_i),
$$

其中

$$
\xi_i \sim U(-1,1),
\qquad
\alpha = \texttt{noise\_level}.
$$

噪声为

$$
\eta_i = g_i^\delta - g_i = \alpha g_i \xi_i.
$$

Morozov 需要估计

$$
\delta = \|\eta\|_2.
$$

---

## 3. 保守上界方法

因为 $|\xi_i| \le 1$，所以

$$
|\eta_i| \le \alpha |g_i|.
$$

因此

$$
\|\eta\|_2 \le \alpha \|g\|_2.
$$

但 $g$ 未知。由三角不等式可得

$$
\|g^\delta\|_2
=
\|g+\eta\|_2
\ge
\|g\|_2 - \|\eta\|_2.
$$

又因为

$$
\|\eta\|_2 \le \alpha\|g\|_2,
$$

所以

$$
\|g^\delta\|_2
\ge
(1-\alpha)\|g\|_2.
$$

当 $0 \le \alpha < 1$ 时，得到

$$
\|g\|_2
\le
\frac{1}{1-\alpha}\|g^\delta\|_2.
$$

代回噪声上界：

$$
\|\eta\|_2
\le
\frac{\alpha}{1-\alpha}\|g^\delta\|_2.
$$

因此保守半径取为

$$
\boxed{
\delta_{\mathrm{cons}}
=
\frac{\alpha}{1-\alpha}\|g^\delta\|_2
}.
$$

### 特点

- 不使用 `g_clean`；
- 是严格保守上界；
- 通常会高估真实噪声；
- Morozov 约束会更松，TV 重建更平滑；
- 可能导致系数相对误差 `coeff_res` 变大。

当 $\alpha=0.1$ 时，系数为

$$
\frac{\alpha}{1-\alpha}
=
\frac{0.1}{0.9}
\approx 0.1111.
$$

---

## 4. RMS 典型噪声估计方法

保守上界考虑最坏情况 $|\xi_i|=1$。但当前噪声是均匀随机噪声，典型大小可用均方/RMS 估计。

因为

$$
\xi_i \sim U(-1,1),
$$

所以

$$
E[\xi_i]=0,
\qquad
E[\xi_i^2]=\frac{1}{3}.
$$

噪声能量期望为

$$
E\|\eta\|_2^2
=
E\sum_i \alpha^2 g_i^2\xi_i^2
=
\frac{\alpha^2}{3}\|g\|_2^2.
$$

因此典型噪声大小约为

$$
\|\eta\|_2
\approx
\frac{\alpha}{\sqrt{3}}\|g\|_2.
$$

但 $g$ 未知，需要用 $g^\delta$ 替代。由

$$
g_i^\delta = g_i(1+\alpha\xi_i),
$$

有

$$
\|g^\delta\|_2^2
=
\sum_i g_i^2(1+\alpha\xi_i)^2.
$$

展开得

$$
\|g^\delta\|_2^2
=
\sum_i g_i^2
\left(1+2\alpha\xi_i+\alpha^2\xi_i^2\right).
$$

取期望：

$$
E\|g^\delta\|_2^2
=
\left(1+\frac{\alpha^2}{3}\right)\|g\|_2^2.
$$

因此

$$
\|g\|_2
\approx
\frac{\|g^\delta\|_2}{\sqrt{1+\alpha^2/3}}.
$$

代回

$$
\|\eta\|_2
\approx
\frac{\alpha}{\sqrt{3}}\|g\|_2,
$$

得到

$$
\delta_{\mathrm{rms}}
\approx
\frac{\alpha}{\sqrt{3}}
\cdot
\frac{\|g^\delta\|_2}{\sqrt{1+\alpha^2/3}}.
$$

整理为

$$
\boxed{
\delta_{\mathrm{rms}}
\approx
\frac{\alpha}{\sqrt{3+\alpha^2}}\|g^\delta\|_2
}.
$$

### 特点

- 不使用 `g_clean`；
- 利用 $\xi_i \sim U(-1,1)$ 的统计信息；
- 更接近平均意义下的真实噪声；
- 不是严格上界，可能低估单次样本中的真实噪声；
- 在当前随机均匀乘性噪声实验中，通常比保守上界更接近旧的 oracle Morozov 结果。

当 $\alpha=0.1$ 时，系数为

$$
\frac{\alpha}{\sqrt{3+\alpha^2}}
=
\frac{0.1}{\sqrt{3.01}}
\approx 0.05764.
$$

---

## 5. 两种方法对比

| 方法 | 噪声半径 | 是否用 `g_clean` | 是否严格上界 | 倾向 |
|---|---:|---:|---:|---|
| Oracle 旧方法 | $\|g^\delta-g\|_2$ | 是 | 是真实噪声 | 只能用于仿真评估，不应作为算法输入 |
| 保守上界 | $\dfrac{\alpha}{1-\alpha}\|g^\delta\|_2$ | 否 | 是 | 更平滑，可能欠拟合 |
| RMS 估计 | $\dfrac{\alpha}{\sqrt{3+\alpha^2}}\|g^\delta\|_2$ | 否 | 否 | 更接近平均真实噪声 |

---

## 6. selected16 TV ADMM=80 实验对比

实验设置：

- 角度文件：`data/alpha_search_cache/alpha_selected16.json`
- 方法：TV constrained Morozov
- ADMM 外层迭代：80
- 乘性噪声：$\alpha=0.1$
- 评价中的 `coeff_res` 为系数 $c_k$ 的相对误差：

$$
\mathrm{coeff\_res}
=
\frac{\|c_{\mathrm{est}}-c_{\mathrm{true}}\|_2}{\|c_{\mathrm{true}}\|_2}.
$$

| 方案 | Morozov 选取值 $\delta$ | 测量残差 $\|Ac-g^\delta\|_2$ | `coeff_res` | TV 值 |
|---|---:|---:|---:|---:|
| 旧 oracle 方法 | 526.526550 | 525.880310 | 0.099687 | 955.243652 |
| 观测保守上界 | 1015.409607 | 1003.064819 | 0.199287 | 832.593140 |
| 观测 RMS 估计 | 526.745117 | 525.919250 | 0.099712 | 954.084351 |

可以看到：

- 保守上界的 $\delta$ 接近 oracle 的 1.93 倍，因此约束明显放宽，重建更平滑，`coeff_res` 明显变大；
- RMS 估计的 $\delta$ 与 oracle 非常接近，因此重建误差也非常接近旧结果；
- RMS 估计仍然不使用 `g_clean`，但它依赖噪声分布假设 $\xi_i \sim U(-1,1)$。

---

## 7. 建议

如果目标是严格满足“噪声上界已知”的 Morozov 理论表述，使用保守方法：

$$
\delta_{\mathrm{cons}}
=
\frac{\alpha}{1-\alpha}\|g^\delta\|_2.
$$

如果目标是在当前仿真乘性均匀噪声模型下获得更接近 oracle Morozov 的效果，同时不把 `g_clean` 作为输入，使用 RMS 方法：

$$
\delta_{\mathrm{rms}}
\approx
\frac{\alpha}{\sqrt{3+\alpha^2}}\|g^\delta\|_2.
$$

报告中建议明确写成：

> 本文不使用无噪声观测 $g$ 或真实系数作为 Morozov 选参输入。对于乘性均匀噪声，实验采用基于观测 $g^\delta$ 的噪声半径估计。保守版本给出严格上界，RMS 版本利用噪声分布的二阶矩估计典型噪声大小。
