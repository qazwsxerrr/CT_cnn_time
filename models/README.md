# models 目录说明

当前 `ct_time/models` 只保留一条主链：

- 系数空间：`B1*B1`
- 图像尺寸：`128x128`
- 角度数：`8`
- 训练数据：`random_ellipses`
- 验证/测试数据：`shepp_logan`
- 训练入口：`train.py`
- 测试入口：`test.py`

## 关键文件

- [config.py](/D:/ai_code/ai_project/ct_time/models/config.py)
  - 主配置
  - 默认数据源与噪声模式
  - 模型、训练、日志路径

- [model.py](/D:/ai_code/ai_project/ct_time/models/model.py)
  - 当前 learned optimizer
  - 多角度梯度以 `per-angle channels` 形式输入网络

- [radon_transform.py](/D:/ai_code/ai_project/ct_time/models/radon_transform.py)
  - `B1*B1` 时域算子
  - `random_ellipses` / `shepp_logan` 数据生成
  - `Tikhonov` 初始化

- [train.py](/D:/ai_code/ai_project/ct_time/models/train.py)
  - 在线训练入口
  - 训练目标固定为 `RES`

- [test.py](/D:/ai_code/ai_project/ct_time/models/test.py)
  - 加载当前 checkpoint
  - 输出 `Mean RES(init)` 和 `Mean RES(pred)`

## 当前默认配置

- `train_data_source = random_ellipses`
- `val_data_source = shepp_logan`
- `test_data_source = shepp_logan`
- `noise_mode = additive`
- `noise_level = 0.1`
- `target_snr_db = 30.0` when `noise_mode = snr`
- `lambda_select_mode = fixed`
- `init_method = cg` or `tikhonov_direct`
- `regularizer_type = tikhonov`

## 运行方式

训练：

```powershell
$env:N_TRAIN_OVERRIDE='1'
$env:N_DATA_OVERRIDE='1'
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\train.py"
```

测试：

```powershell
& "D:\python_code\anaconda_mini\envs\pytorch_env\python.exe" "D:\ai_code\ai_project\ct_time\models\test.py"
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
- 日志：
  - [training.log](/D:/ai_code/ai_project/ct_time/logs/training.log)
  - [training_progress.png](/D:/ai_code/ai_project/ct_time/logs/training_progress.png)
- 结果图：
  - [results](/D:/ai_code/ai_project/ct_time/results)

## 备注

- `tikhonov_eval.py` 仍保留比较型评估逻辑，因此会继续包含 `B1*B1/B2*B1` 相关代码。
- 当前主链不再依赖旧实验的数据生成、旧损失函数或旧多角度融合分支。
