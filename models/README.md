# models 目录说明

当前 `ct_time/models` 只保留一条主链，并在同一套代码上支持多个 8/16 角实验 profile：

- 系数空间：`B1*B1`
- 图像尺寸：`128x128`
- 默认角度配置：`8` 个固定特定角
- 可切换到 `16` 角实验 profile
- 训练数据：`random_ellipses`
- 验证/测试数据：`shepp_logan`
- 训练入口：`train.py`
- 测试入口：`test.py`

## 本次 README 修改说明

- 修正了 README 中已经过时的默认配置描述（如噪声模式、`lambda_select_mode`、初始化方式）。
- 补充了两个新的 16 角实验运行方式：
  - 实验 A：`8 特定 + 8 自动单射角`
  - 实验 B：`16 自动单射角，全 16 角下三角`
- 增加了当前推荐的最佳 16 角主链 profile：
  - `injective16_pi_best`
- 增加了 experiment profile 的说明，方便直接按 tag 训练、评估和对比。

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

- [train.py](/D:/ai_code/ai_project/ct_time/models/train.py)
  - 在线训练入口
  - 训练目标固定为 `RES`

- [test.py](/D:/ai_code/ai_project/ct_time/models/test.py)
  - 加载当前 checkpoint
  - 输出 `Mean RES(init)` 和 `Mean RES(pred)`

## 实验 profile 概览

- `default`
  - 默认 8 特定角主链
  - 当前配置文件启动时的默认 profile

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

- `experiment_profile = default`
- `train_data_source = random_ellipses`
- `val_data_source = shepp_logan`
- `test_data_source = shepp_logan`
- `operator_mode = theoretical_b1b1`
- `num_angles_total = 8`
- `multi_angle_layout = structured_backbone_extra`
- `multi_angle_solver_mode = stacked_tikhonov`
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
- Tikhonov 初始化走 `split_triangular_admm`
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
