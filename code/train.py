import torch.nn as nn
import torch.optim as optim
import csv
import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from datetime import timedelta


class TimeDataset(Dataset):
    """带时间字符串的 Dataset，返回 (X, y, time_str)。"""
    def __init__(self, X, y, times):
        self.X = X
        self.y = y
        self.times = times

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.times[idx]

from config import DS_PLUS

# Legacy constants kept exactly as used for the reported runs.
# NOTE: main.py uses (5150 - 3641)/100*(-0.71); see NOTES.md before changing.
dif = (5100 - 3641) / 100 * (-0.72)
df = pd.read_csv(DS_PLUS, parse_dates=['valid_time'])


def func(d, a, b):
    return a * (d ** b)


def get_prior_water_event_batch(df, event_times, ddf=3.44, dry_threshold=0.1,
                                max_lookback_hours=24, dif=dif):
    """批量处理一整个 batch 的 event_times，返回供水信息。"""
    start_times, durations, total_waters = [], [], []
    for event_time in event_times:
        event_time = pd.to_datetime(event_time)
        past = df[df['valid_time'] <= event_time].copy()
        if len(past) == 0:
            print('数据缺失')
        cutoff = event_time - timedelta(hours=max_lookback_hours)
        past = past[past['valid_time'] > cutoff]
        past_desc = past.sort_values('valid_time', ascending=False)
        start_time = event_time
        duration_h = 0
        total_water_mm = 0
        for _, row in past_desc.iterrows():
            rain = row['rainfall_mm'] if row['rainfall_mm'] > dry_threshold else 0
            temp_corrected = row['temprature_Cel'] + dif
            melt = max(temp_corrected, 0) * (ddf / 24.0)
            hourly_water = rain + melt
            if hourly_water > 0:
                total_water_mm += hourly_water
                duration_h += 1
                start_time = row['valid_time']
            else:
                break
        start_times.append(start_time)
        durations.append(duration_h)
        total_waters.append(round(total_water_mm, 2))
    return start_times, durations, total_waters


def _compute_p_batch(batch_times_strs, curve_a, curve_b):
    """计算 batch 中每个样本的物理先验概率 p。"""



    import math
    start_times, durations, waters = get_prior_water_event_batch(
        df=df, event_times=batch_times_strs)
    p_batch = []
    for d, w in zip(durations, waters):
        if w > 0 and d > 0:
            w_pred = curve_a * (d ** curve_b)
            ratio = w / w_pred
            # p = 1.0 / (1.0 + math.exp(-k * (ratio - 1.0)))
            p=max(0,min(ratio,1))
        else:
            p = 0.0  # 无产水 → 物理上不可能发生泥石流
        p_batch.append(p)
    return torch.tensor(p_batch, dtype=torch.float32)


def _train_one_epoch(model, loader, optimizer, criterion, pinn, curve_a, curve_b, curve_r, device):
    """训练一个 epoch，返回 (avg_loss, avg_acc)。"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    for batch_x, batch_y, batch_times_strs in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x).squeeze()

        if pinn:
            p_tensor = _compute_p_batch(batch_times_strs, curve_a, curve_b).to(device)

            # 逐样本损失 (reduction='none')
            loss_data = nn.BCEWithLogitsLoss(reduction='none')(outputs.view(-1), batch_y.view(-1))
            # loss_phys = nn.BCEWithLogitsLoss(reduction='none')(outputs.view(-1), p_tensor.view(-1))
            loss_phys=nn.MSELoss(reduction='none')(torch.sigmoid(outputs).view(-1), p_tensor.view(-1))

            # 物理先验与标签一致的样本: 有产水(p>0) 且 (p>0.5 & y=1) 或 (p<0.5 & y=0)
            mask_agree = (p_tensor > 0.0) & (
                ((p_tensor > 0.5) & (batch_y == 1)) |
                ((p_tensor < 0.5) & (batch_y == 0))
            )

            #满足边界条件时 采取混合损失
            lam = min(max(curve_r, 0.0), 1.0)
            loss = torch.where(mask_agree,
                               (1.0 - lam) * loss_data + lam * loss_phys,
                               loss_data).mean()

            #修改PINN的写法 不采取混合损失
            # loss = torch.where(mask_agree, loss_phys, loss_data).mean()

            #直接检验loss_phys行不行 效果差
            # loss=loss_phys.mean()

        else:
            loss = criterion(outputs.view(-1), batch_y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = batch_x.size(0)
        total_loss += loss.item() * n
        pred_labels = (torch.sigmoid(outputs) >= 0.5).float()
        total_acc += (pred_labels == batch_y).sum().item()
        total_samples += n

    return total_loss / total_samples, total_acc / total_samples


@torch.no_grad()
def _eval_loader(model, loader, criterion, device):
    """在 loader 上评估，返回 (avg_loss, avg_acc)。"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    for batch_x, batch_y, _ in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = model(batch_x).squeeze()
        loss = criterion(outputs.view(-1), batch_y.view(-1))
        n = batch_x.size(0)
        total_loss += loss.item() * n
        pred_labels = (torch.sigmoid(outputs) > 0.5).float()
        total_acc += (pred_labels == batch_y).sum().item()
        total_samples += n
    return total_loss / total_samples, total_acc / total_samples


def _train_single_fold(model, train_loader, val_loader, epochs, lr, weight_decay,
                        pinn, curve_a, curve_b, curve_r, patience, device,
                        save_dir, model_name, fold_idx):
    """
    训练单个 fold：train_loader 训练，val_loader 做 early stopping。
    返回 (best_model_state, best_val_loss, train_log, val_log, best_epoch)。
    """
    os.makedirs(save_dir, exist_ok=True)
    pth = os.path.join(save_dir, f'{model_name}_fold{fold_idx}.pth')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # CosineAnnealing: T_max = epochs → 起始 lr 到 0 的完整周期
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Warmup: 前 warmup_epochs 轮线性增长
    warmup_epochs = 5
    base_lr = lr

    best_val_loss = float('inf')
    best_epoch = -1
    best_state = None
    patience_count = 0

    train_log = {'loss': [], 'acc': []}
    val_log = {'loss': [], 'acc': []}

    for epoch in range(epochs):
        # Warmup
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * (epoch + 1) / warmup_epochs

        # 训练（含 dropout）
        avg_train_loss, avg_train_acc = _train_one_epoch(
            model, train_loader, optimizer, criterion, pinn,
            curve_a, curve_b, curve_r, device)
        train_log['loss'].append(avg_train_loss)
        train_log['acc'].append(avg_train_acc)

        scheduler.step()

        # 验证（关闭 dropout）
        avg_val_loss, avg_val_acc = _eval_loader(model, val_loader, criterion, device)
        val_log['loss'].append(avg_val_loss)
        val_log['acc'].append(avg_val_acc)

        # checkpoint：val_loss 最低
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_count = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), pth)
        else:
            patience_count += 1

        marker = ' *' if best_epoch == epoch + 1 else ''
        print(f"  Fold[{fold_idx}] Epoch [{epoch+1:4d}/{epochs}]{marker} | "
              f"Train L:{avg_train_loss:.4f} A:{avg_train_acc:.4f} | "
              f"Val L:{avg_val_loss:.4f} A:{avg_val_acc:.4f} | patience:{patience_count}/{patience}")

        if patience_count >= patience:
            print(f"  >> Fold[{fold_idx}] early stop at epoch {epoch+1}: val_loss 不创新低 {patience} 轮")
            break

    # 加载最佳权重
    model.load_state_dict(best_state)
    return best_state, best_val_loss, train_log, val_log, best_epoch


def train_model(model, train_loader, test_loader, epochs=100, lr=1e-3, weight_decay=1e-4,
                PINN=False, idx=-1, curve_a=0.238, curve_b=1.265, curve_r=0.841,
                save_dir='mnt', model_name=None, k_folds=5):
    """
    K-Fold 交叉验证训练 LSTM 模型。

    固定训练协议（与论文一致；早期版本中的“自适应”设置已弃用）：
      K   = 5      StratifiedKFold，仅在训练划分内部切分
      ep  = 100    上限，配合 early stopping
      pat = 10     val_loss 连续 10 轮不下降即停止

    流程：
      - 训练集 → StratifiedKFold(K=5) → train_sub + val_sub
      - val_sub 做 early stopping
      - 选取 val_loss 最低的折作为最终模型
      - 测试集只评估一次，不参与任何选择
    """
    device = next(model.parameters()).device
    os.makedirs(save_dir, exist_ok=True)

    if model_name is None:
        model_name = 'PI_lstm' if PINN else 'lstm'

    # ---- 从 train_loader 提取全部训练数据 ----
    all_X, all_y, all_times = [], [], []
    for bx, by, bt in train_loader:
        all_X.append(bx)
        all_y.append(by)
        all_times.extend(bt)
    X_train_all = torch.cat(all_X, dim=0)
    y_train_all = torch.cat(all_y, dim=0)
    times_all = all_times

    n_train = len(y_train_all)
    n_pos = int(y_train_all.sum().item())
    n_neg = n_train - n_pos

    # ---- 固定训练协议（可被命令行覆盖，仅用于调试/冒烟测试）----
    k_folds = int(k_folds)
    epochs = int(epochs)
    PATIENCE = 10

    print(f"\nK-Fold CV (K={k_folds}) | 训练集: {n_train} 样本 (正:{n_pos} 负:{n_neg})")

    # ---- Stratified K-Fold ----
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=3407 + idx)

    # 初始权重备份（每折重新初始化）
    initial_state = copy.deepcopy(model.state_dict())

    cv_train_acc, cv_val_acc = [], []
    cv_train_loss, cv_val_loss = [], []
    best_fold_idx = -1
    best_fold_val_loss = float('inf')
    best_fold_state = None
    best_fold_train_log = None
    best_fold_val_log = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_all.cpu().numpy(),
                                                                y_train_all.cpu().numpy())):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx+1}/{k_folds} | train: {len(train_idx)}  val: {len(val_idx)} "
              f"(val_pos={int(y_train_all[val_idx].sum().item())})")
        print(f"{'='*50}")

        # 重置模型权重
        model.load_state_dict(copy.deepcopy(initial_state))

        # 构建 fold 的 train/val DataLoader（带时间字符串，供 PINN 使用）
        fold_times_train = [times_all[i] for i in train_idx]
        fold_times_val   = [times_all[i] for i in val_idx]

        fold_train_ds = TimeDataset(X_train_all[train_idx], y_train_all[train_idx], fold_times_train)
        fold_val_ds   = TimeDataset(X_train_all[val_idx], y_train_all[val_idx], fold_times_val)

        fold_train_loader = DataLoader(fold_train_ds, batch_size=8, shuffle=True,
                                       num_workers=0, pin_memory=True)
        fold_val_loader = DataLoader(fold_val_ds, batch_size=8, shuffle=False,
                                     num_workers=0, pin_memory=True)

        # 训练单折
        best_state, best_val_loss, train_log, val_log, best_epoch = _train_single_fold(
            model=model, train_loader=fold_train_loader, val_loader=fold_val_loader,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            pinn=PINN, curve_a=curve_a, curve_b=curve_b, curve_r=curve_r,
            patience=PATIENCE, device=device,
            save_dir=save_dir, model_name=model_name, fold_idx=fold_idx)

        # 记录本折最佳指标
        best_train_acc = train_log['acc'][best_epoch - 1] if best_epoch > 0 else train_log['acc'][-1]
        best_val_acc = val_log['acc'][best_epoch - 1] if best_epoch > 0 else val_log['acc'][-1]
        cv_train_acc.append(best_train_acc)
        cv_val_acc.append(best_val_acc)
        cv_train_loss.append(train_log['loss'][best_epoch - 1] if best_epoch > 0 else train_log['loss'][-1])
        cv_val_loss.append(best_val_loss)

        print(f"  Fold {fold_idx} best: val_loss={best_val_loss:.4f} val_acc={best_val_acc:.4f} "
              f"train_acc={best_train_acc:.4f} @ epoch {best_epoch}")

        if best_val_loss < best_fold_val_loss:
            best_fold_val_loss = best_val_loss
            best_fold_idx = fold_idx
            best_fold_state = best_state
            best_fold_train_log = train_log
            best_fold_val_log = val_log

    # ---- 加载最佳折的模型 ----
    model.load_state_dict(best_fold_state)
    final_pth = os.path.join(save_dir, f'{model_name}{idx}.pth')
    torch.save(best_fold_state, final_pth)

    # ---- 汇总 CV 结果 ----
    train_acc_mean = np.mean(cv_train_acc)
    train_acc_std = np.std(cv_train_acc, ddof=1) if len(cv_train_acc) > 1 else 0.0
    val_acc_mean = np.mean(cv_val_acc)
    val_acc_std = np.std(cv_val_acc, ddof=1) if len(cv_val_acc) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"K-Fold CV 汇总 (K={k_folds})")
    print(f"{'='*60}")
    print(f"  CV Train Acc: {train_acc_mean:.4f} +/- {train_acc_std:.4f}")
    print(f"  CV Val   Acc: {val_acc_mean:.4f} +/- {val_acc_std:.4f}")
    print(f"  最佳折: Fold {best_fold_idx} (val_loss={best_fold_val_loss:.4f})")
    print(f"{'='*60}")

    # ---- 在 test_loader 上评估（记录参考值，不参与选择）----
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc = _eval_loader(model, test_loader, criterion, device)
    print(f"  Test (held-out, reference): Loss={test_loss:.4f} Acc={test_acc:.4f}")

    # ---- 构建兼容返回的 log（使用最佳折的 log，补充 CV 均值）----
    # main.py 访问 train_log['acc'][-1] → 用最佳折的最后值
    train_log = best_fold_train_log if best_fold_train_log else {'loss': [], 'acc': []}
    val_log = best_fold_val_log if best_fold_val_log else {'loss': [], 'acc': []}

    # 保存图片
    plot_training_curve(train_log, val_log,
                        best_epoch=len(train_log['loss']),
                        save_dir=save_dir, model_name=model_name, idx=idx,
                        ed_a=curve_a, ed_b=curve_b, ed_r=curve_r)

    # ---- 保存 CSV ----
    csv_file = os.path.join(save_dir, 'res', f'{model_name}.csv')
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_exists = os.path.exists(csv_file)

    # CSV 列：train_acc(最佳折) | test_acc(held-out, 参考) | cv_train_mean | cv_val_mean | ed_a | ed_b | ed_r2
    row = [
        round(train_log['acc'][-1], 4) if train_log['acc'] else np.nan,
        round(test_acc, 4),
        round(train_acc_mean, 4),
        round(val_acc_mean, 4),
        round(curve_a, 3),
        round(curve_b, 3),
        round(curve_r, 3),
    ]
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["train_acc", "test_acc", "cv_train_acc_mean",
                             "cv_val_acc_mean", "ed_a", "ed_b", "ed_r2"])
        writer.writerow(row)
    print(f"  Saved: {csv_file} | train_acc={row[0]:.4f} test_acc={row[1]:.4f} "
          f"cv_train={row[2]:.4f} cv_val={row[3]:.4f}")

    cv_metrics = {
        'cv_train_acc_mean': round(train_acc_mean, 4),
        'cv_train_acc_std': round(train_acc_std, 4),
        'cv_val_acc_mean': round(val_acc_mean, 4),
        'cv_val_acc_std': round(val_acc_std, 4),
        'cv_val_loss_mean': round(np.mean(cv_val_loss), 4),
        'train_val_gap': round(train_acc_mean - val_acc_mean, 4),
    }

    return model, train_log, val_log, cv_metrics


def plot_training_curve(train_log, val_log, best_epoch=-1, save_dir='mnt',
                        model_name='model', idx=0, ed_a=0, ed_b=0, ed_r=0):
    """绘制训练曲线，保存到 curves/ 子目录"""
    curves_dir = os.path.join(save_dir, 'curves')
    os.makedirs(curves_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    n_epochs = len(train_log['loss'])

    # 损失曲线
    ax1.plot(train_log['loss'], label='Train', color='#E74C3C', linewidth=2)
    ax1.plot(val_log['loss'], label='Val', color='#3498DB', linewidth=2)
    if 0 < best_epoch <= n_epochs:
        ax1.axvline(best_epoch, color='#2ca02c', ls='--', lw=1, alpha=0.7,
                    label=f'Best (ep {best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name}{idx} — Loss  (ED: a={ed_a:.2f} b={ed_b:.2f} R$^2$={ed_r:.2f})',
                  fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # 准确率曲线
    ax2.plot(train_log['acc'], label='Train', color='#E74C3C', linewidth=2)
    ax2.plot(val_log['acc'], label='Val', color='#3498DB', linewidth=2)
    if 0 < best_epoch <= n_epochs:
        ax2.axvline(best_epoch, color='#2ca02c', ls='--', lw=1, alpha=0.7,
                    label=f'Best (ep {best_epoch})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{model_name}{idx} — Accuracy', fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(curves_dir, f'{model_name}{idx}.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()


if __name__ == '__main__':
    print('train_test')
    print(nn.BCEWithLogitsLoss()(torch.tensor(-0.1), torch.tensor(1.0)))
