import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import time
import os
import sys
import logging
import matplotlib
import random
import numpy as np
import multiprocessing  # 显式导入用于检测进程

# 强制使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# [Windows 多进程防刷屏补丁] - 必须在导入 config 之前执行
# ============================================================================
if os.name == 'nt':
    # 检查当前是否为主进程
    if multiprocessing.current_process().name != 'MainProcess':
        # 如果是子进程 (DataLoader workers)，禁止打印到 stdout/stderr
        class NullWriter:
            def write(self, s): pass

            def flush(self): pass


        sys.stdout = NullWriter()
        sys.stderr = NullWriter()

# 引入项目配置
# 注意：这些导入必须在上面的补丁之后，因为 config.py 里面有 print 语句
from model import initialize_model
from config import (
    n_data,
    device, MODEL_PATH, BEST_MODEL_PATH, CHECKPOINT_DIR,
    TRAINING_CONFIG, LOGGING_CONFIG, DATA_DIR
)
from train_offline import CTDataset, tv_loss, seed_everything

# ============================================================================
# 配置区域
# ============================================================================
EXTRA_ITERATIONS = 2500
# 梯度累积步数：3

ACCUMULATION_STEPS = 3

# 微调时的初始学习率（如果无法加载优化器状态时使用）
FINETUNE_LR = 1e-4

# 验证频率 (每 10 个累积步验证一次，即每 40 个物理 batch)
VALIDATION_INTERVAL = 10


class ContinueTrainer:
    def __init__(self):
        self._setup_logging()
        seed_everything(42)

        self.model = initialize_model()
        self.scaler = torch.amp.GradScaler('cuda')

        train_path = os.path.join(DATA_DIR, "train_dataset.pt")
        val_path = os.path.join(DATA_DIR, "val_dataset.pt")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing training data: {train_path}")

        train_dataset = CTDataset(train_path)

        # 训练集加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=n_data,  # 物理 Batch Size 保持 10 不变
            shuffle=True,  # Shuffle 必须开启，保证梯度累积的多样性
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        if os.path.exists(val_path):
            val_dataset = CTDataset(val_path)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=n_data,
                shuffle=False,  # 验证集保持固定顺序，确保可比性
                num_workers=2,
                pin_memory=True
            )
        else:
            self.val_loader = None

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAINING_CONFIG["optimizer_learning_rate"],
            weight_decay=1e-4
        )

        lr_lambda = lambda step: 1.0 / (1.0 + step / 500.0)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.start_iter = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_res': [], 'val_res': [],
            'learning_rate': [], 'data_fidelity_error': []
        }

        self._load_checkpoint()
        self.criterion_l1 = nn.L1Loss()
        self.target_iter = self.start_iter + EXTRA_ITERATIONS

        self.logger.info(f"Resuming training from iter {self.start_iter}")
        self.logger.info(f"Target iteration: {self.target_iter}")
        self.logger.info(
            f"Gradient Accumulation Steps: {ACCUMULATION_STEPS} (Effective Batch Size: {n_data * ACCUMULATION_STEPS})")
        self.logger.info(f"Validation Interval: Every {VALIDATION_INTERVAL} accumulated steps")

    def _setup_logging(self):
        log_dir = LOGGING_CONFIG['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger('ContinueTrainer')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - [CONTINUE] - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(os.path.join(log_dir, 'training_continue.log'))
        file_handler.setFormatter(console_formatter)
        self.logger.addHandler(file_handler)

    def _load_checkpoint(self):
        load_path = BEST_MODEL_PATH
        if not os.path.exists(load_path):
            self.logger.warning(f"Best model not found at {load_path}, trying standard model path...")
            load_path = MODEL_PATH
            if not os.path.exists(load_path):
                raise FileNotFoundError("No model found to continue training!")

        self.logger.info(f"Loading checkpoint from: {load_path}")
        checkpoint = torch.load(load_path, map_location=device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info("Model weights loaded successfully.")

        if 'iter' in checkpoint: self.start_iter = checkpoint['iter']
        if 'best_val_loss' in checkpoint: self.best_val_loss = checkpoint['best_val_loss']
        if 'training_history' in checkpoint: self.training_history = checkpoint['training_history']

        try:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("Optimizer state restored.")
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        except Exception as e:
            self.logger.warning(f"Could not restore optimizer state (error: {e}). Resetting optimizer.")
            self.optimizer = optim.AdamW(self.model.parameters(), lr=FINETUNE_LR, weight_decay=1e-4)
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0)

    def _validate(self):
        if self.val_loader is None:
            return 0.0, 0.0, {}

        self.model.eval()
        total_val_loss = 0.0
        total_val_res = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, (coeff_true, g_observed, coeff_initial) in enumerate(self.val_loader):
                coeff_true = coeff_true.to(device, non_blocking=True)
                g_observed = g_observed.to(device, non_blocking=True)
                coeff_initial = coeff_initial.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    coeff_pred, _, metrics = self.model(coeff_initial, g_observed)

                    diff_sq_sum = torch.sum(torch.abs(coeff_pred - coeff_true) ** 2)
                    true_sq_sum = torch.sum(torch.abs(coeff_true) ** 2)
                    val_res = torch.sqrt(diff_sq_sum / true_sq_sum)

                    l1_val = self.criterion_l1(coeff_pred, coeff_true)
                    tv_val = tv_loss(coeff_pred)
                    val_loss = l1_val + 0.1 * tv_val

                total_val_loss += val_loss.item()
                total_val_res += val_res.item()
                num_batches += 1
                if i >= 10: break

        self.model.train()
        return total_val_loss / num_batches, total_val_res / num_batches, metrics

    def _save_plots(self):
        if len(self.training_history['train_loss']) == 0: return
        try:
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 1, 1)
            plt.plot(self.training_history['train_loss'], label='Train Loss')
            plt.axvline(x=self.start_iter, color='r', linestyle='--', label='Resume Point')
            plt.title('Loss History')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(self.training_history['train_res'], label='Train RES')
            plt.axvline(x=self.start_iter, color='r', linestyle='--', label='Resume Point')
            plt.title('Residual Error History')
            plt.legend()

            plt.savefig(os.path.join(LOGGING_CONFIG['log_dir'], 'continue_training_plot.png'))
            plt.close()
        except Exception:
            pass

    def train(self):
        self.logger.info("Starting CONTINUATION training with Gradient Accumulation...")
        self.model.train()

        current_iter = self.start_iter
        patience_counter = 0

        self.optimizer.zero_grad()

        accumulated_loss_for_log = 0.0
        accumulated_res_for_log = 0.0

        # 循环直到达到目标 iter
        while current_iter < self.target_iter:
            for i, (coeff_true, g_observed, coeff_initial) in enumerate(self.train_loader):
                if current_iter >= self.target_iter: break

                # --- 1. 数据准备 ---
                coeff_true = coeff_true.to(device)
                g_observed = g_observed.to(device)
                coeff_initial = coeff_initial.to(device)

                # --- 2. 前向传播 ---
                with torch.amp.autocast('cuda'):
                    coeff_pred, _, metrics = self.model(coeff_initial, g_observed)

                    l1_val = self.criterion_l1(coeff_pred, coeff_true)
                    tv_val = tv_loss(coeff_pred)
                    loss = l1_val + 0.1 * tv_val

                    # [关键] Loss 除以累积步数，保证梯度量级正确
                    loss = loss / ACCUMULATION_STEPS

                    with torch.no_grad():
                        diff = torch.sum(torch.abs(coeff_pred - coeff_true) ** 2)
                        true_s = torch.sum(torch.abs(coeff_true) ** 2)
                        res_error = torch.sqrt(diff / true_s)

                # --- 3. 反向传播 (攒梯度) ---
                self.scaler.scale(loss).backward()

                accumulated_loss_for_log += loss.item() * ACCUMULATION_STEPS
                accumulated_res_for_log += res_error.item()

                # --- 4. 真正更新参数 (每 4 个 batch 更新一次) ---
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # 计算平滑后的 Loss/Res
                    avg_loss = accumulated_loss_for_log / ACCUMULATION_STEPS
                    avg_res = accumulated_res_for_log / ACCUMULATION_STEPS

                    self.training_history['train_loss'].append(avg_loss)
                    self.training_history['train_res'].append(avg_res)
                    self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

                    accumulated_loss_for_log = 0.0
                    accumulated_res_for_log = 0.0

                    current_iter += 1

                    # --- [修改] 验证逻辑 (每 10 次有效更新验证一次) ---
                    if current_iter % VALIDATION_INTERVAL == 0:
                        val_loss, val_res, _ = self._validate()
                        self.training_history['val_loss'].append(val_loss)
                        self.training_history['val_res'].append(val_res)

                        log_msg = (f"Iter {current_iter}/{self.target_iter} | "
                                   f"Loss: {avg_loss:.5f} | "
                                   f"Train RES: {avg_res:.4f} | "
                                   f"Val RES: {val_res:.4f} | "
                                   f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            patience_counter = 0
                            self._save_checkpoint(current_iter, is_best=True)
                            self.logger.info(log_msg + " [BEST SAVED]")
                        else:
                            patience_counter += 1
                            self.logger.info(log_msg)

                        self._save_checkpoint(current_iter, is_best=False)
                        self._save_plots()

                    if patience_counter > 500:
                        self.logger.info("Early stopping triggered.")
                        return

        self._save_checkpoint(current_iter, is_best=False)
        self.logger.info("Continuation training finished.")

    def _save_checkpoint(self, current_iter, is_best=False):
        state = {
            'iter': current_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        torch.save(state, MODEL_PATH)
        if is_best:
            torch.save(state, BEST_MODEL_PATH)


if __name__ == '__main__':
    trainer = ContinueTrainer()
    trainer.train()
