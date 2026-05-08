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
- 网络物理角度数、learned operator 角度数、CNN per-angle gradient 通道数自动与 JSON 中的 α 数量一致
- Tikhonov 初始化作为 `coeff_current` 初始值输入网络，不额外添加 Tikhonov 图像通道
- α continuous operator 是 full sparse block，不使用下三角残差通道或下三角显式更新

### 16 个 α 角度训练

```powershell
$env:EXPERIMENT_PROFILE_OVERRIDE = "alpha_condition"
$env:ALPHA_CONDITION_JSON_OVERRIDE = "D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_selected16.json"
$env:OUTPUT_TAG_OVERRIDE = "alpha16_diag_deep"

& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" `
  "D:\ai_code\ai_project\ct_time\models\deep_learn\train.py"
```

### 16 个 α 角度、排除窗口版本

```powershell
$env:EXPERIMENT_PROFILE_OVERRIDE = "alpha_condition"
$env:ALPHA_CONDITION_JSON_OVERRIDE = "D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_selected16_exclude0.3.json"
$env:OUTPUT_TAG_OVERRIDE = "alpha16_exclude_diag_deep"

& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" `
  "D:\ai_code\ai_project\ct_time\models\deep_learn\train.py"
```

### 快速 smoke 训练

```powershell
$env:EXPERIMENT_PROFILE_OVERRIDE = "alpha_condition"
$env:ALPHA_CONDITION_JSON_OVERRIDE = "D:\ai_code\ai_project\ct_time\data\alpha_search_cache\alpha_selected16.json"
$env:OUTPUT_TAG_OVERRIDE = "alpha16_smoke"
$env:N_TRAIN_OVERRIDE = "1"
$env:N_DATA_OVERRIDE = "1"

& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" `
  "D:\ai_code\ai_project\ct_time\models\deep_learn\train.py"
```

## 输出位置

- 模型与训练 checkpoint：`checkpoints/deep_learn/`，文件名会带 `OUTPUT_TAG_OVERRIDE`。
- 日志：`logs/<OUTPUT_TAG_OVERRIDE>/`
- 角度选择与 Tikhonov 评估不在本目录运行，见 `models/α_condition/README.md`。
