# deep_learn：神经网络重建链路

本目录只保留神经网络模型、训练与测试入口。公共配置、算子、图像生成与离线数据生成入口统一放在 `models/` 顶层。

## 文件职责

```text
models/deep_learn/
  model.py             # Learned optimizer / CNN 模型
  train.py             # 训练入口
  test.py              # 测试入口
  README.md
```

顶层统一文件为：

```text
models/config.py          # 公共配置 + 神经网络训练/测试配置
models/radon_transform.py # 前向/伴随算子、Tikhonov 初始化、数据生成器
models/image_generator.py # 图像/phantom 生成
models/Data_Generator.py  # 可选：离线数据生成入口
```

`models/α_condition` 与 `models/deep_learn` 都从顶层这些文件读取依赖，避免两套 `config.py` / `radon_transform.py` 分叉。

## 当前推荐 profile：α 条件数采样

- `EXPERIMENT_PROFILE_OVERRIDE=alpha_condition`
- 角度 JSON 来自 `data/alpha_search_cache/alpha_selected{K}.json`
- 网络物理角度数、learned operator 角度数自动与 JSON 中的 α 数量一致
- CNN 输入角度通道可用 `CNN_ANGLE_INDICES_OVERRIDE` 从全部 α 中选择子集；当前推荐 16 个物理角度中均匀选择 8 个 CNN 输入角度
- Tikhonov 初始化作为 `coeff_current` 初始值输入网络，不额外添加 Tikhonov 图像通道
- α continuous operator 是 full sparse block，不使用 β 整数投影、整数排序、下三角构造、下三角残差通道或下三角显式更新
- `PHYSICS_RESIDUAL_MODE_OVERRIDE=per_angle_cg` 表示每个 α 角度单独构造一个 physics residual 通道；若设置了 `CNN_ANGLE_INDICES_OVERRIDE`，数据保真项梯度通道和 physics residual 通道会使用同一组角度索引
- 测试脚本中的 `Mean RES (tikhonov)` 是重新计算的纯 Tikhonov baseline，默认使用 `EVAL_TIKHONOV_BASELINE_METHOD_OVERRIDE=tikhonov_direct`

## 训练调用方法

```powershell
Set-Location "D:\ai_code\ai_project\ct_time"

$env:EXPERIMENT_PROFILE_OVERRIDE = "alpha_condition"
$env:ALPHA_CONDITION_JSON_OVERRIDE = "D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_selected16.json"
$env:ALPHA_GRAM_CACHE_DIR_OVERRIDE = "D:\ai_code\ai_project\ct_time\data\alpha_gram_cache"

# 16 个物理角度中均匀选 8 个角度作为 CNN 输入：
# - 8 个 per-angle 数据保真项梯度通道
# - 8 个对应 per-angle_cg physics residual 通道
$env:CNN_ANGLE_INDICES_OVERRIDE = "0,2,4,6,8,10,12,14"
$env:CNN_NUM_ANGLES_OVERRIDE = "8"

# Tikhonov/Morozov 初始化
$env:INIT_METHOD_OVERRIDE = "tikhonov_direct"
$env:LAMBDA_SELECT_MODE_OVERRIDE = "morozov"

# 噪声设置
$env:NOISE_MODE_OVERRIDE = "multiplicative"
$env:NOISE_LEVEL_OVERRIDE = "0.1"

# per-angle physics residual 通道
$env:PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE = "1"
$env:PHYSICS_RESIDUAL_MODE_OVERRIDE = "per_angle_cg"
$env:PHYSICS_RESIDUAL_DAMPING_OVERRIDE = "1e-2"
$env:PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE = "8"
$env:PHYSICS_RESIDUAL_DETACH_OVERRIDE = "1"
$env:PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE = "1"

# 可选：显式 physics update；当前 alpha16_8 推荐关闭
$env:PHYSICS_EXPLICIT_UPDATE_ENABLED_OVERRIDE = "0"

# 中间监督
$env:INTERMEDIATE_SUPERVISION_ENABLED_OVERRIDE = "1"
$env:INTERMEDIATE_SUPERVISION_WEIGHT_START_OVERRIDE = "0.2"
$env:INTERMEDIATE_SUPERVISION_WEIGHT_END_OVERRIDE = "1.0"

# 训练规模与输出 tag
$env:N_TRAIN_OVERRIDE = "5000"
$env:N_DATA_OVERRIDE = "8"
$env:OUTPUT_TAG_OVERRIDE = "alpha16_even8_grad_phys_morozov_direct_noise01"

& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" `
  "D:\ai_code\ai_project\ct_time\models\deep_learn\train.py"
```

输出位置：

- 最优模型：`D:\ai_code\ai_project\ct_time\checkpoints\deep_learn\theoretical_ct_alpha16_even8_grad_phys_morozov_direct_noise01_best_model.pth`
- 最终模型：`D:\ai_code\ai_project\ct_time\checkpoints\deep_learn\theoretical_ct_alpha16_even8_grad_phys_morozov_direct_noise01_model.pth`
- 训练日志：`D:\ai_code\ai_project\ct_time\logs\alpha16_even8_grad_phys_morozov_direct_noise01\training.log`
- 中间 checkpoint：`D:\ai_code\ai_project\ct_time\checkpoints\deep_learn\checkpoints_alpha16_even8_grad_phys_morozov_direct_noise01\`

如果希望直接保存到自定义 `logs/alpha16_8` 之类目录，需要在代码里改 `OUTPUT_TAG_OVERRIDE` 对应的 tag，或手动移动/重命名 checkpoint；当前默认路径由 `models/config.py` 中的 `MODEL_DIR` 和 `LOG_DIR` 统一控制。

### 快速 smoke 训练

用于只验证配置、模型初始化、Morozov cache 与一次训练迭代是否能跑通：

```powershell
Set-Location "D:\ai_code\ai_project\ct_time"

$env:EXPERIMENT_PROFILE_OVERRIDE = "alpha_condition"
$env:ALPHA_CONDITION_JSON_OVERRIDE = "D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_selected16.json"
$env:ALPHA_GRAM_CACHE_DIR_OVERRIDE = "D:\ai_code\ai_project\ct_time\data\alpha_gram_cache"
$env:CNN_ANGLE_INDICES_OVERRIDE = "0,2,4,6,8,10,12,14"
$env:CNN_NUM_ANGLES_OVERRIDE = "8"
$env:INIT_METHOD_OVERRIDE = "tikhonov_direct"
$env:LAMBDA_SELECT_MODE_OVERRIDE = "morozov"
$env:NOISE_MODE_OVERRIDE = "multiplicative"
$env:NOISE_LEVEL_OVERRIDE = "0.1"
$env:PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE = "1"
$env:PHYSICS_RESIDUAL_MODE_OVERRIDE = "per_angle_cg"
$env:PHYSICS_RESIDUAL_DAMPING_OVERRIDE = "1e-2"
$env:PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE = "8"
$env:PHYSICS_RESIDUAL_DETACH_OVERRIDE = "1"
$env:PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE = "1"
$env:INTERMEDIATE_SUPERVISION_ENABLED_OVERRIDE = "1"
$env:INTERMEDIATE_SUPERVISION_WEIGHT_START_OVERRIDE = "0.2"
$env:INTERMEDIATE_SUPERVISION_WEIGHT_END_OVERRIDE = "1.0"
$env:N_TRAIN_OVERRIDE = "1"
$env:N_DATA_OVERRIDE = "1"
$env:OUTPUT_TAG_OVERRIDE = "alpha16_even8_smoke"

& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" `
  "D:\ai_code\ai_project\ct_time\models\deep_learn\train.py"
```

## 测试调用方法

测试时必须保证模型结构配置和训练 checkpoint 一致，尤其是：

- `ALPHA_CONDITION_JSON_OVERRIDE`
- `CNN_ANGLE_INDICES_OVERRIDE`
- `CNN_NUM_ANGLES_OVERRIDE`
- `PHYSICS_RESIDUAL_MODE_OVERRIDE`
- `PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE`
- `PHYSICS_RESIDUAL_DAMPING_OVERRIDE`

当前 `test.py` 会优先从 checkpoint metadata 恢复大部分训练配置；但旧 checkpoint 可能没有完整记录 `cnn_angle_indices` / physics residual 字段，因此推荐测试时仍显式设置下面这些环境变量。

```powershell
Set-Location "D:\ai_code\ai_project\ct_time"

$env:EXPERIMENT_PROFILE_OVERRIDE = "alpha_condition"
$env:ALPHA_CONDITION_JSON_OVERRIDE = "D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_selected16.json"
$env:ALPHA_GRAM_CACHE_DIR_OVERRIDE = "D:\ai_code\ai_project\ct_time\data\alpha_gram_cache"

$env:CNN_ANGLE_INDICES_OVERRIDE = "0,2,4,6,8,10,12,14"
$env:CNN_NUM_ANGLES_OVERRIDE = "8"

$env:INIT_METHOD_OVERRIDE = "tikhonov_direct"
$env:LAMBDA_SELECT_MODE_OVERRIDE = "morozov"
$env:NOISE_MODE_OVERRIDE = "multiplicative"
$env:NOISE_LEVEL_OVERRIDE = "0.1"
$env:TEST_DATA_SOURCE_OVERRIDE = "shepp_logan"

$env:PHYSICS_RESIDUAL_CHANNEL_ENABLED_OVERRIDE = "1"
$env:PHYSICS_RESIDUAL_MODE_OVERRIDE = "per_angle_cg"
$env:PHYSICS_RESIDUAL_DAMPING_OVERRIDE = "1e-2"
$env:PHYSICS_RESIDUAL_CG_ITERS_OVERRIDE = "8"
$env:PHYSICS_RESIDUAL_DETACH_OVERRIDE = "1"
$env:PHYSICS_RESIDUAL_NORMALIZE_OVERRIDE = "1"

# 测试报告中的 Tikhonov baseline，默认就是 tikhonov_direct；这里显式写出便于复现实验
$env:EVAL_TIKHONOV_BASELINE_METHOD_OVERRIDE = "tikhonov_direct"

& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" `
  "D:\ai_code\ai_project\ct_time\models\deep_learn\test.py" `
  --model-path "D:\ai_code\ai_project\ct_time\logs\alpha16_8\theoretical_ct_alpha16_even8_grad_phys_morozov_direct_noise01_best_model.pth" `
  --num-samples 50 `
  --result-dir "D:\ai_code\ai_project\ct_time\results\alpha16_8_eval" `
  --result-prefix "alpha16_even8_grad_phys_morozov_direct_noise01"
```

如果模型按默认 `OUTPUT_TAG_OVERRIDE` 保存在 `checkpoints/deep_learn/`，则 `--model-path` 可改成：

```powershell
--model-path "D:\ai_code\ai_project\ct_time\checkpoints\deep_learn\theoretical_ct_alpha16_even8_grad_phys_morozov_direct_noise01_best_model.pth"
```

测试输出会打印：

```text
Mean RES (tikhonov): ...
Mean RES (pred): ...
```

其中：

- `Mean RES (tikhonov)`：同一观测、同一 Morozov $\lambda$ 下重新计算的纯 Tikhonov baseline
- `Mean RES (pred)`：神经网络输出结果

## 目录约定

- 模型与训练 checkpoint：`D:\ai_code\ai_project\ct_time\checkpoints\deep_learn\`
- 日志：`D:\ai_code\ai_project\ct_time\logs\<OUTPUT_TAG_OVERRIDE>\`
- 数据与缓存：`D:\ai_code\ai_project\ct_time\data\`
- 测试/评估结果：`D:\ai_code\ai_project\ct_time\results\`
- 角度选择与 Tikhonov 评估不在本目录运行，见 `models/α_condition/README.md`。
