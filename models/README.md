# models 目录说明

当前 `ct_time/models` 只保留一条主链，并在同一套代码上支持多个 8/16 角实验 profile，以及配套的条件数/采样方案分析脚本：

- 系数空间：`B1*B1`
- 图像尺寸：`128x128`
- 兼容 `8` 角 backbone 与 `16` 角 full-triangular 两条实验口径
- 可切换到 `16` 角实验 profile
- 训练数据：`random_ellipses`
- 验证/测试数据：`shepp_logan`
- 训练入口：`train.py`
- 测试入口：`test.py`

## 当前仓库内容

- 主训练/测试链路：
  - `config.py`
  - `model.py`
  - `radon_transform.py`
  - `train.py`
  - `test.py`
- 初始化与评估辅助：
  - `morozov_init_eval.py`
  - `tikhonov_eval.py`
  - `tikhonov_find.py`
- 条件数与采样方案分析：
  - `compare_legacyext_sampling_condition_numbers.py`
  - `plot_beta_k_equals_t_unit_square_compare.py`

## 关键文件

- [config.py](/D:/ai_code/ai_project/ct_time/models/config.py)
  - 主配置
  - 默认数据源与噪声模式
  - 模型、训练、日志路径
  - `EXPERIMENT_PROFILE_OVERRIDE` / `OUTPUT_TAG_OVERRIDE` 等运行时入口

- [model.py](/D:/ai_code/ai_project/ct_time/models/model.py)
  - 当前 learned optimizer
  - 多角度梯度以 `per-angle channels` 形式输入网络

- [radon_transform.py](/D:/ai_code/ai_project/ct_time/models/radon_transform.py)
  - `B1*B1` 时域算子
  - `random_ellipses` / `shepp_logan` 数据生成
  - `Tikhonov` 初始化
  - 单射排序下三角构造
  - `split_triangular_admm`
  - `legacy / shifted_support / legacy_injective_extension` 三种公式口径
  - 导出每角的 `support_lo_beta` / `theory_t0_abs` 等分析元数据

- [train.py](/D:/ai_code/ai_project/ct_time/models/train.py)
  - 在线训练入口
  - 训练目标固定为 `RES`

- [test.py](/D:/ai_code/ai_project/ct_time/models/test.py)
  - 加载当前 checkpoint
  - 输出 `Mean RES(init)` 和 `Mean RES(pred)`

- [compare_legacyext_sampling_condition_numbers.py](/D:/ai_code/ai_project/ct_time/models/compare_legacyext_sampling_condition_numbers.py)
  - 对当前 `injective16_pi_best` 16 角集合逐角比较
    - `legacy_injective_extension`
    - `sampling`
  - 计算整矩阵条件数、带宽、对角值等统计
  - 输出 `.json` / `.md` 分析结果

- [plot_beta_k_equals_t_unit_square_compare.py](/D:/ai_code/ai_project/ct_time/models/plot_beta_k_equals_t_unit_square_compare.py)
  - 画出 `beta·k=t` 在线族与单位正方形中的几何示意
  - 用于解释 `t=0.5` 与 `t=A_beta+0.5` 两类采样规则差异

## 实验 profile 概览

- `default`
  - 保留的兼容 profile
  - 对应旧的 8 特定角主链

- `structured16_injective_extra`
  - 实验 A
  - `8 特定 + 8 自动单射角`
  - 这 `16` 个角全部构造成下三角系统
  - 其中前 `8` 个角固定为原先的特定角
  - 学习阶段也使用全部 `16` 角

- `injective16_full_triangular`
  - 实验 B
  - 自动挑选 `16` 个满足单射条件的 `β`
  - `16` 个角全部构造成下三角系统
  - 初始化走 `split_triangular_admm`
  - 学习阶段也使用全部 `16` 角

- `injective16_pi_best`
  - 当前推荐主链
  - 直接固定为效果最好的 `(0,\pi)` 口径下的 `16` 个单射角
  - `16` 个角全部构造成下三角系统
  - Tikhonov 初始化、learned optimizer 与 CNN 通道全部统一为 `16`

## 当前默认配置

- `experiment_profile = injective16_pi_best`
- `train_data_source = random_ellipses`
- `val_data_source = shepp_logan`
- `test_data_source = shepp_logan`
- `operator_mode = theoretical_b1b1`
- `num_angles_total = 16`
- `multi_angle_layout = full_triangular`
- `multi_angle_solver_mode = stacked_tikhonov`
- `theoretical_formula_mode = legacy_injective_extension`
- `noise_mode = multiplicative`
- `noise_level = 0.1`
- `target_snr_db = 30.0` when `noise_mode = snr`
- `lambda_select_mode = morozov`
- `init_method = tikhonov_direct`
- `regularizer_type = tikhonov`

## 推荐主链：最佳 16 角 `(0,\pi)` 方案

如果要直接使用当前效果最好的 16 角主链，推荐使用：

```powershell
$env:EXPERIMENT_PROFILE_OVERRIDE = "injective16_pi_best"
$env:GLOBAL_SEED_OVERRIDE = "20260327"
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\train.py"
```

该 profile 的特点：

- 使用固定的最佳 `16` 个可下三角化角度
- 初始化求解走 `stacked_tikhonov`
- 单角块构造使用 `legacy_injective_extension`
- learned optimizer 使用全部 `16` 个物理角
- CNN 通道数固定为 `16`

## 运行方式

进入目录：

```powershell
cd D:\ai_code\ai_project\ct_time\models
```

如果切换过实验 profile，建议先清理环境变量：

```powershell
Remove-Item Env:EXPERIMENT_PROFILE_OVERRIDE -ErrorAction SilentlyContinue
Remove-Item Env:OUTPUT_TAG_OVERRIDE -ErrorAction SilentlyContinue
Remove-Item Env:BETA_VECTORS_OVERRIDE -ErrorAction SilentlyContinue
Remove-Item Env:NUM_ANGLES_TOTAL_OVERRIDE -ErrorAction SilentlyContinue
Remove-Item Env:EXPLICIT_EXTRA_BETA_VECTORS_OVERRIDE -ErrorAction SilentlyContinue
```

默认训练（建议用于快速检查）：

训练：

```powershell
$env:N_TRAIN_OVERRIDE='1'
$env:N_DATA_OVERRIDE='1'
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\train.py"
```

默认测试：

测试：

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\test.py"
```

### 实验 A：8 特定 + 8 自动单射角

```powershell
$env:EXPERIMENT_PROFILE_OVERRIDE = "structured16_injective_extra"
$env:GLOBAL_SEED_OVERRIDE = "20260327"
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\train.py"
```

对应输出 tag 默认是：

- `structured16_injective_extra`

说明：

- 这 `16` 个角会全部走下三角化与 `split_triangular_admm`
- 只是其中前 `8` 个角固定包含原先的 8 个特定角

对应 best checkpoint：

- `D:\ai_code\ai_project\ct_time\models\theoretical_ct_structured16_injective_extra_best_model.pth`

### 实验 B：16 自动单射角，全 16 角下三角

```powershell
$env:EXPERIMENT_PROFILE_OVERRIDE = "injective16_full_triangular"
$env:GLOBAL_SEED_OVERRIDE = "20260327"
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\train.py"
```

对应输出 tag 默认是：

- `injective16_full_triangular`

说明：

- 这 `16` 个角也全部走下三角化与 `split_triangular_admm`
- 与实验 A 的区别仅在于角集合不强制包含原先那 `8` 个特定角

对应 best checkpoint：

- `D:\ai_code\ai_project\ct_time\models\theoretical_ct_injective16_full_triangular_best_model.pth`

### 评估实验 A / B

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\test.py" --compare-tags structured16_injective_extra,injective16_full_triangular --num-samples 50
```

### 评估推荐的最佳 16 角主链

```powershell
$env:EXPERIMENT_PROFILE_OVERRIDE = "injective16_pi_best"
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\test.py" --model-path "D:\ai_code\ai_project\ct_time\models\theoretical_ct_injective16_pi_best_best_model.pth" --num-samples 50
```

### 条件数分析：legacy_injective_extension vs sampling

在 `models/` 目录下运行：

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\compare_legacyext_sampling_condition_numbers.py"
```

默认会分析当前 `injective16_pi_best` 这组 16 个角，并生成：

- `legacyext_vs_sampling_condition_numbers_injective16_pi_best.json`
- `legacyext_vs_sampling_condition_numbers_injective16_pi_best.md`

如果只想快速查看帮助：

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\compare_legacyext_sampling_condition_numbers.py" --help
```

### 采样几何示意图

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\plot_beta_k_equals_t_unit_square_compare.py"
```

默认输出：

- `D:\ai_code\ai_project\ct_time\models\beta_k_equals_t_unit_square_compare.svg`

### 与旧的 8特定+8随机 对比

如果旧实验 tag 仍使用你之前的：

- `structured16_seed20260327`

则可直接对比：

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\test.py" --compare-tags structured16_seed20260327,structured16_injective_extra,injective16_full_triangular --num-samples 50
```

测试随机椭圆路径：

```powershell
$env:TEST_DATA_SOURCE_OVERRIDE='random_ellipses'
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\test.py"
```

测试 SNR 噪声路径：

```powershell
$env:NOISE_MODE_OVERRIDE='snr'
$env:TARGET_SNR_DB_OVERRIDE='30'
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\test.py"
```

## 产物位置

- 模型：
  - [theoretical_ct_model.pth](/D:/ai_code/ai_project/ct_time/models/theoretical_ct_model.pth)
  - [theoretical_ct_best_model.pth](/D:/ai_code/ai_project/ct_time/models/theoretical_ct_best_model.pth)
  - profile 训练会额外生成：
    - `theoretical_ct_structured16_injective_extra_best_model.pth`
    - `theoretical_ct_injective16_full_triangular_best_model.pth`
    - `theoretical_ct_injective16_pi_best_best_model.pth`
- 日志：
  - [training.log](/D:/ai_code/ai_project/ct_time/logs/training.log)
  - [training_progress.png](/D:/ai_code/ai_project/ct_time/logs/training_progress.png)
  - profile 日志目录：
    - `D:\ai_code\ai_project\ct_time\logs\structured16_injective_extra\`
    - `D:\ai_code\ai_project\ct_time\logs\injective16_full_triangular\`
    - `D:\ai_code\ai_project\ct_time\logs\injective16_pi_best\`
- 结果图：
  - [results](/D:/ai_code/ai_project/ct_time/results)

## 备注

- `tikhonov_eval.py` 仍保留比较型评估逻辑，因此会继续包含 `B1*B1/B2*B1` 相关代码。
- 当前主链不再依赖旧实验的数据生成、旧损失函数或旧多角度融合分支。
- 当前仓库中的日志、权重、临时 `.json/.md` 结果和图片更多作为本地实验产物；上传代码时优先关注 `models/` 下脚本本身。
