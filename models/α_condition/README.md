# α 条件数选角度与 stacked Tikhonov 使用说明

本目录整理了当前 α 连续角度采样相关的两个脚本：

```text
models/α_condition/alpha_condition_constrained_sampling.py
models/α_condition/alpha_tikhonov_eval.py
```

其中：

- `alpha_condition_constrained_sampling.py`：搜索 α 角度与每个角度对应的偏移 $\tau$，以单角度矩阵条件数为主要筛选指标，并用 bucket + beam search 选择最终角度集合。
- `alpha_tikhonov_eval.py`：读取搜索结果，构造 α continuous stacked operator，只执行 Tikhonov 正则化，不接入神经网络。

当前只保留本目录下的入口脚本；旧的顶层兼容入口已清理。

---

## 1. α 条件数搜索的数学形式

对每个角度 $\alpha\in[0,\pi)$，定义单位方向

$$
p_\alpha=(\cos\alpha,\sin\alpha).
$$

对网格点 $k=(k_1,k_2)$ 定义连续投影

$$
s_k(\alpha)=k_1\cos\alpha+k_2\sin\alpha.
$$

排序后得到

$$
s_{(0)}<s_{(1)}<\cdots<s_{(N-1)},
\qquad N=128^2.
$$

对固定 $\tau$，采样点取为

$$
t_i=s_{(i)}+\tau.
$$

对应单角度矩阵为

$$
A_{\alpha,\tau}[i,j]
=R_\alpha\phi\left(s_{(i)}+\tau-s_{(j)}\right).
$$

搜索脚本对每个候选 $\alpha$ 搜索 $\tau$，目标为

$$
\min_\tau \log\operatorname{cond}(A_{\alpha,\tau}).
$$

最后对候选角度做分桶和 beam search，使最终角度既有较小条件数，也尽量覆盖 $[0,\pi)$。

---

## 2. 搜索结果 JSON 结构

搜索输出 JSON 同时包含最终选择和完整搜索记录：

```json
{
  "meta": {...},
  "selected": [...],
  "top8": [...],
  "results": [...]
}
```

字段含义：

- `selected`：最终选出的角度集合，数量由 `--top-k` 决定。
- `top8`：兼容旧配置的别名，实际也保存最终选出的集合。
- `results`：完整搜索记录，包含每一个已评估的候选角度、条件数、$\tau$、奇异值信息等。

单条记录通常包含：

```json
{
  "alpha": 1.570469015841,
  "tau_star": 0.9999262296468905,
  "cond": 38097.14568356,
  "sigma_min": 0.00336,
  "sigma_max": 128.0,
  "lambda_min": ...,
  "lambda_max": ...,
  "min_gap": ...,
  "matrix_nnz": ...,
  "lower_bandwidth": ...,
  "upper_bandwidth": ...,
  "is_valid": true
}
```

---

## 3. 推荐保存位置

建议把搜索结果放在：

```text
D:\ai_code\ai_project\ct_time\data\alpha_search_cache
```

例如：

```text
D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_full_resume.json
```

α stacked Tikhonov 的 Gram 谱缓存建议放在：

```text
D:\ai_code\ai_project\ct_time\data\alpha_gram_cache
```

这两类缓存不同：

- `alpha_search_cache`：保存角度搜索结果 JSON。
- `alpha_gram_cache`：保存 Tikhonov/Morozov 用的 Gram 谱 `.pt` 缓存。

若不显式传 `--output-json`，角度搜索结果默认保存到 `alpha_search_cache`：

- 无排除窗口：`alpha_selected{top_k}.json`，例如 `alpha_selected8.json`
- 有排除窗口：`alpha_selected{top_k}_exclude{exclude_window}.json`，例如 `alpha_selected8_exclude0.3.json`

---

## 4. 运行 α 条件数搜索

### 4.1 创建目录

```powershell
New-Item -ItemType Directory -Force "D:\ai_code\ai_project\ct_time\data\alpha_search_cache"
```

### 4.2 设置线程环境变量

建议外层使用 `--workers` 并行，内层 BLAS 线程设为 1：

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
```

### 4.3 推荐搜索命令

下面例子生成约 3000 个候选角度，并最终选择 16 个角度：

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" `
  "D:\ai_code\ai_project\ct_time\models\α_condition\alpha_condition_constrained_sampling.py" `
  --num-alpha-grid 1000 `
  --num-alpha-random 1000 `
  --num-alpha-golden 1000 `
  --injective-tol 1e-5 `
  --top-k 16 `
  --per-bucket-keep 20 `
  --beam-size 200 `
  --workers 8 `
  --save-every 200
```

此命令默认输出到：

```text
D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_selected16.json
```

### 4.4 断点续跑

脚本支持断点续跑。默认情况下，如果 `--output-json` 已存在，会自动从该 JSON 的 `results` 字段读取已有记录并跳过已完成的 $\alpha$。

因此中断后直接重新运行同一条命令即可。

也可以显式指定：

```powershell
--resume-json "D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_full_resume.json"
```

默认每 200 个新角度保存一次：

```powershell
--save-every 200
```

若想关闭周期保存：

```powershell
--save-every 0
```

### 4.5 多进程进度显示

多进程评估使用 `as_completed`，也就是哪个角度先完成就先打印：

```text
[1/2999] alpha=... cond=... tau=...
[2/2999] alpha=... cond=... tau=...
```

因此日志中的 alpha 不一定按从小到大排列。这是正常现象。

---

## 5. 从已有 results 重新选择 top-k

如果已有完整或部分搜索结果，只想重新选择 8、16、32 个角度，不重新计算条件数，可使用 `--reuse-json`：

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" `
  "D:\ai_code\ai_project\ct_time\models\α_condition\alpha_condition_constrained_sampling.py" `
  --reuse-json "D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_full_resume.json" `
  --top-k 16 `
  --per-bucket-keep 20 `
  --beam-size 200 `
  --output-json "D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_selected16.json"
```

`--reuse-json` 只做选择，不重新评估条件数。

---

## 6. 运行纯 stacked Tikhonov 正则化

`alpha_tikhonov_eval.py` 只运行 Tikhonov，不加载神经网络，不调用 `train.py`。

对选出的角度，构造 stacked operator：

$$
A_{\mathrm{stack}}
=
\begin{bmatrix}
A_{\alpha_1,\tau_1}\\
A_{\alpha_2,\tau_2}\\
\vdots\\
A_{\alpha_K,\tau_K}
\end{bmatrix}.
$$

Tikhonov 解为

$$
(A_{\mathrm{stack}}^TA_{\mathrm{stack}}+\lambda I)c_\lambda
=A_{\mathrm{stack}}^Tg^\delta.
$$

---

## 7. Morozov 选参运行 Tikhonov

若搜索时使用 `--top-k 16`，这里也应使用 `--top-k 16`：

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" `
  "D:\ai_code\ai_project\ct_time\models\α_condition\alpha_tikhonov_eval.py" `
  --top-k 16 `
  --scenario all `
  --lambda-mode morozov `
  --data-source shepp_logan `
  --num-trials 1 `
  --alpha-gram-cache-dir "D:\ai_code\ai_project\ct_time\data\alpha_gram_cache"
```

若不传 `--alpha-json`，默认读取：

```text
D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_selected16.json
```

如果要读取带排除窗口的默认角度文件，可在 Tikhonov 命令中同样传 `--exclude-window`，例如 `--exclude-window 0.3` 会默认读取：

```text
D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_selected16_exclude0.3.json
```

若不传 `--output-prefix` 与 `--output-dir`，默认结果目录与文件前缀为：

```text
D:\ai_code\ai_project\ct_time\results\alpha16_diag_morozov
alpha16_diag_morozov_results.json
alpha16_diag_morozov_results.txt
alpha16_diag_morozov_mult_0_1.png
...
```

若角度 JSON 文件名或元信息包含排除窗口，例如 `alpha_selected16_exclude0.3.json`，默认前缀会变为：

```text
alpha16_exclude_diag_morozov
```

输出重点看：

```text
lambda
noise_norm
measurement_residual
coeff_res
```

其中 `coeff_res` 是 Tikhonov 重建系数的相对误差。

---

## 8. 固定 λ 运行 Tikhonov

如果想先避免 Morozov 选参耗时，可固定 $\lambda$：

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" `
  "D:\ai_code\ai_project\ct_time\models\α_condition\alpha_tikhonov_eval.py" `
  --top-k 16 `
  --scenario all `
  --lambda-mode fixed `
  --lambda-reg 0.01 `
  --data-source shepp_logan `
  --num-trials 1 `
  --alpha-gram-cache-dir "D:\ai_code\ai_project\ct_time\data\alpha_gram_cache"
```

---

## 9. 场景选择

`--scenario` 支持：

```text
all
mult_0_1
mult_0_05
mult_0_04
mult_0_03
mult_0_02
mult_0_01
```

含义：

- `mult_0_1`：乘性噪声强度 0.1。
- `mult_0_05`：乘性噪声强度 0.05。
- `mult_0_04`：乘性噪声强度 0.04。
- `mult_0_03`：乘性噪声强度 0.03。
- `mult_0_02`：乘性噪声强度 0.02。
- `mult_0_01`：乘性噪声强度 0.01。
- `all`：依次运行以上六个场景。

---

## 10. 输出文件

例如使用：

```text
--output-prefix alpha16_tikhonov_morozov
--output-dir D:\ai_code\ai_project\ct_time\results\alpha16_tikhonov_morozov
```

会生成：

```text
alpha16_tikhonov_morozov_results.json
alpha16_tikhonov_morozov_results.txt
alpha16_tikhonov_morozov_mult_0_1.png
alpha16_tikhonov_morozov_mult_0_05.png
alpha16_tikhonov_morozov_mult_0_04.png
alpha16_tikhonov_morozov_mult_0_03.png
alpha16_tikhonov_morozov_mult_0_02.png
alpha16_tikhonov_morozov_mult_0_01.png
```

---

## 11. 注意事项

1. 第一次运行 Tikhonov 会构造 Gram 谱缓存，可能较慢；后续同一组角度和 $\tau$ 会复用缓存。
2. `--top-k` 应与搜索 JSON 中想使用的角度数量一致。
3. 如果 JSON 中已有 `selected`，脚本优先使用 `selected`；若只有 `results`，则会按 `--top-k` 重新做 bucket + beam 选择。
4. 多进程搜索时 alpha 输出顺序不再代表角度大小顺序，而代表完成顺序。
5. 搜索得到的是单角度条件数较好的候选；最终 Tikhonov 使用的是 stacked operator，其恢复效果应以 `coeff_res` 为准。
