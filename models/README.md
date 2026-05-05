# models 目录说明

当前 `ct_time/models` 已收敛到**两种采样/矩阵构造**，用于做干净的对比实验：

1. **条件数采样**
   - profile：`condition_constrained8_pi`
   - formula：`condition_constrained_offset`
   - 矩阵：**完整 sparse**
2. **下三角采样**
   - profile：`same8_shifted_support_triangular_pi`
   - formula：`legacy_injective_extension`
   - 矩阵：**exact lower-banded 下三角**

两条链路都使用：

- 系数空间：`B1*B1`
- 图像尺寸：`128x128`
- 训练入口：`train.py`
- 测试入口：`test.py`
- 初始化求解：`stacked_tikhonov`
- 数据生成规则：**观测数据必须使用完整合法算子，不允许用“自洽但缺失上三角”的作弊式生成**

---

## 当前主链文件

- `config.py`
  - profile 选择
  - 运行时 override
  - 输出路径与训练/测试配置
- `radon_transform.py`
  - 两种保留采样方式的算子构造
  - 干净观测数据生成
  - `Morozov + Tikhonov` 初始化链路
- `model.py`
  - learned optimizer
- `train.py`
  - 训练入口
- `test.py`
  - 测试与对比入口

辅助分析脚本：

- `compare_legacyext_sampling_condition_numbers.py`
  - 比较两种采样下的条件数与矩阵统计
- `plot_beta_k_equals_t_unit_square_compare.py`
  - 画采样几何示意图

---

## 两个实验 profile

### 1）条件数采样

- `EXPERIMENT_PROFILE_OVERRIDE=condition_constrained8_pi`
- `theoretical_formula_mode=condition_constrained_offset`
- `multi_angle_layout=full_triangular`
- `multi_angle_solver_mode=stacked_tikhonov`

说明：

- 使用 `best_condition_constrained8_pi.json` 中的 8 个 `β`
- 每个角使用对应的 `tau_star`
- 观测/重建都走**完整 sparse 矩阵**
- 不会丢掉上三角观测项

### 2）下三角采样

- `EXPERIMENT_PROFILE_OVERRIDE=same8_shifted_support_triangular_pi`
- `theoretical_formula_mode=legacy_injective_extension`
- `multi_angle_layout=full_triangular`
- `multi_angle_solver_mode=stacked_tikhonov`

说明：

- 使用与 `condition_constrained8_pi` 相同的 8 个 `β`
- 采样点按下三角单射构造
- 对应矩阵本身就是**exact lower-banded 下三角**
- 这条链路允许带状存储，因为它不是“伪下三角”

---

## 当前默认配置

默认 profile：

- `experiment_profile = condition_constrained8_pi`

其余关键默认值：

- `train_data_source = random_ellipses`
- `val_data_source = shepp_logan`
- `test_data_source = shepp_logan`
- `operator_mode = theoretical_b1b1`
- `multi_angle_layout = full_triangular`
- `multi_angle_solver_mode = stacked_tikhonov`
- `theoretical_formula_mode = condition_constrained_offset`
- `data_formula_mode = auto_complete`
- `lambda_select_mode = morozov`
- `init_method = tikhonov_direct`

---

## 运行方式

进入目录：

```powershell
cd D:\ai_code\ai_project\ct_time\models
```

如需切换 profile，建议先清理旧环境变量：

```powershell
Remove-Item Env:EXPERIMENT_PROFILE_OVERRIDE -ErrorAction SilentlyContinue
Remove-Item Env:OUTPUT_TAG_OVERRIDE -ErrorAction SilentlyContinue
Remove-Item Env:BETA_VECTORS_OVERRIDE -ErrorAction SilentlyContinue
Remove-Item Env:NUM_ANGLES_TOTAL_OVERRIDE -ErrorAction SilentlyContinue
```

### 默认训练

```powershell
$env:N_TRAIN_OVERRIDE='1'
$env:N_DATA_OVERRIDE='1'
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\train.py"
```

### 默认测试

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\test.py"
```

### 跑条件数采样训练

```powershell
$env:EXPERIMENT_PROFILE_OVERRIDE = "condition_constrained8_pi"
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\train.py"
```

### 跑下三角采样训练

```powershell
$env:EXPERIMENT_PROFILE_OVERRIDE = "same8_shifted_support_triangular_pi"
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\train.py"
```

### 对比两个 tag

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\test.py" --compare-tags condition_constrained8_pi,same8_shifted_support_triangular_pi --num-samples 50
```

---

## 条件数与采样分析

### 比较两种采样的条件数统计

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\compare_legacyext_sampling_condition_numbers.py"
```

### 采样几何示意图

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\plot_beta_k_equals_t_unit_square_compare.py"
```

---

## 产物位置

- 模型：
  - `models/theoretical_ct_condition_constrained8_pi_model.pth`
  - `models/theoretical_ct_condition_constrained8_pi_best_model.pth`
  - `models/theoretical_ct_same8_shifted_support_triangular_pi_model.pth`
  - `models/theoretical_ct_same8_shifted_support_triangular_pi_best_model.pth`
- 日志：
  - `logs/condition_constrained8_pi/`
  - `logs/same8_shifted_support_triangular_pi/`
- 结果：
  - `results/`

---

## 备注

- 当前主链已经删除旧的 `16` 角 profile、`split_triangular_admm`、`consensus_admm`、`structured_backbone_extra` 等历史入口。
- 如果后续要做“条件数采样 vs 下三角采样”的严格对比，应始终确认：
  - `condition_constrained8_pi` 使用**完整 sparse**
  - `same8_shifted_support_triangular_pi` 使用**exact lower-banded**
  - 加噪前的干净观测由对应的**完整合法前向算子**生成
