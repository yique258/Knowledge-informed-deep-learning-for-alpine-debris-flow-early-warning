import data_process
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from data_process import nowater_event
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from model import LSTM
from config import RESULT_DIR

def load_best_model(model_path="lstm/mnt/best_LSTM1_debris_flow.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(
        input_size=2,
        hidden_size=16,
        num_layers=2,
        dropout=0.2,
        output_size=1
    ).to(device)  # 放到GPU（不可用时回落到CPU）

    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ 成功加载最佳模型：{model_path}")
    return model

def calculate_regression_metrics(y_true, y_pred):
    """
    计算 R², MSE, RMSE, MAE
    y_true: 真实标签 (numpy array)
    y_pred: 预测概率 (numpy array)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    return {
        'R2': round(r2, 4),
        'MSE': round(mse, 6),
        'RMSE': round(rmse, 6),
        'MAE': round(mae, 6)
    }
def evaluate_model(model, dataloader, criterion=None, is_val=False):
    """评估模型：计算损失、准确率、AUC，重点分析无产水样本"""
    model.eval()  # 评估模式（关闭Dropout）
    device = next(model.parameters()).device
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    all_pred_probs = []  # 存储预测概率（用于AUC计算）
    all_true_labels = []  # 存储真实标签

    with torch.no_grad():  # 禁用梯度计算（加速+防内存泄露）
        for batch_x, batch_y,_ in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            outputs = outputs.squeeze()

            # 计算损失（验证集需要，测试集可选）
            if criterion is not None:
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)

            # 计算准确率
            pred_labels = (torch.sigmoid(outputs)>= 0.5).float()
            total_acc += (pred_labels == batch_y).sum().item()

            # 收集概率和真实标签（用于AUC）
            all_pred_probs.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            all_true_labels.extend(batch_y.cpu().numpy().flatten())

            total_samples += batch_x.size(0)

    # 计算平均指标
    avg_loss = total_loss / total_samples if criterion is not None else 0.0
    avg_acc = total_acc / total_samples
    avg_auc = roc_auc_score(all_true_labels, all_pred_probs) if len(set(all_true_labels)) > 1 else 0.0
    metrics = calculate_regression_metrics(all_true_labels, all_pred_probs)
    # 非验证集（测试集）需打印详细评估
    if not is_val:
        print(f"\n" + "=" * 60)
        print("测试集评估结果")
        print("=" * 60)
        print(f"准确率（Accuracy）：{avg_acc:.4f}")
        print(f"AUC值：{avg_auc:.4f}")
        print(f"\n分类报告：")
        print(classification_report(
            np.array(all_true_labels),
            (np.array(all_pred_probs) >= 0.5).astype(int),
            target_names=['无泥石流', '泥石流']
        ))
        print(f"Val Loss: {avg_loss:.4f} | "
              f"R²: {metrics['R2']:.4f} | "
              f"MSE: {metrics['MSE']:.6f} | "
              f"RMSE: {metrics['RMSE']:.6f} | "
              f"MAE: {metrics['MAE']:.6f}")

    return avg_loss, avg_acc, all_pred_probs, all_true_labels


def analyze_no_rain_samples(sample_info_test, y_test, y_pred_probs):
    """分析测试集中无产水泥石流样本的预测效果"""

    no_water_mask = sample_info_test['time'].isin(nowater_event)
    no_water_info = sample_info_test.loc[no_water_mask].copy()

    no_rain_y_true = y_test[no_water_mask]
    no_rain_y_pred_probs = np.array(y_pred_probs)[no_water_mask]
    no_rain_y_pred = (no_rain_y_pred_probs >= 0.5).astype(int)

    # 生成分析结果
    no_rain_result = pd.DataFrame({
        '事件时间': no_water_info['time'].dt.strftime('%Y-%m-%d %H:%M'),
        '真实标签': no_rain_y_true.astype(int),
        '预测概率': no_rain_y_pred_probs.round(4),
        '预测标签': no_rain_y_pred,
        '预测是否正确': (no_rain_y_true.astype(int) == no_rain_y_pred).astype(str)
    })

    print(f"\n" + "=" * 60)
    print("无产水泥石流样本预测详情")
    print("=" * 60)
    print(no_rain_result.to_string(index=False))

    # 准确率
    no_rain_acc = accuracy_score(no_rain_y_true, no_rain_y_pred)
    print(
        f"\n无产水泥石流样本预测准确率：{no_rain_acc:.4f}（{sum(no_rain_y_true == no_rain_y_pred)}/{len(no_rain_y_true)}）")

    # 完整结果
    all_result = pd.DataFrame({
        '事件时间': sample_info_test['time'].dt.strftime('%Y-%m-%d %H:%M'),
        '样本类型': sample_info_test.apply(
            lambda x: '无产水泥石流' if x['time'] in nowater_event else (
                '普通泥石流' if x['label'] == 1 else '无泥石流'),
            axis=1
        ),
        '真实标签': y_test.astype(int),
        '预测概率': np.array(y_pred_probs).round(4),
        '预测标签': (np.array(y_pred_probs) >= 0.5).astype(int),
        '预测是否正确': (y_test.astype(int) == (np.array(y_pred_probs) >= 0.5).astype(int)).astype(str)
    })

    out_path = RESULT_DIR / 'lstm_prediction_results.csv'
    all_result.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"\n完整预测结果已保存：{out_path}")

    return no_rain_result, all_result


if __name__ == '__main__':
    print('evaluate')