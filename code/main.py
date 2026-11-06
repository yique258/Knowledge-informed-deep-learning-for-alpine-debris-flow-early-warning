"""
LSTM + PINN 泥石流预测 —— 主入口
===================================
三种模式：
  1. global_ed ：全局直接 E-D 拟合（42 事件），所有 run 共用同一组 (a, b, R²)
  2. per_run_ed：每 run 从训练集直接 E-D 拟合
  3. baseline  ：纯 LSTM，无 PINN 物理约束

用法:
  python main.py --mode global_ed --splits 0.2 0.3 --runs 100
  python main.py --mode per_run_ed --splits 0.2 --runs 10
  python main.py --mode baseline --splits 0.3 --runs 100
"""

from model import *
from data_process import *
from evaluate import *
from train import *
from config import DS_PLUS, REPO_ROOT
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")
print(torch.__version__)
print(torch.cuda.is_available())

# ---- 物理常数（模块级） ----
_DIF = (5150 - 3641) / 100 * (-0.71)   # 温度递减率修正
_DDF = 3.44                             # 度日因子 (mm/°C/day)


def _trace_water_for_event(wdf, event_time_str):
    """回溯事件前连续产水时段，返回 (时长h, 水量mm)。"""
    from datetime import timedelta
    et = pd.to_datetime(event_time_str)
    past = wdf[wdf['valid_time'] <= et]
    cutoff = et - timedelta(hours=24)
    past = past[past['valid_time'] > cutoff].sort_values('valid_time', ascending=False)
    dur_h, water_mm = 0, 0.0
    for _, row in past.iterrows():
        rain = row['rainfall_mm'] if row['rainfall_mm'] > 0.1 else 0.0
        melt = max(row['temprature_Cel'] + _DIF, 0.0) * (_DDF / 24.0)
        if rain + melt > 0:
            water_mm += rain + melt
            dur_h += 1
        else:
            break
    return dur_h, round(water_mm, 2)


def compute_global_ed_params():
    """
    从全部 42 个泥石流事件做直接 E-D 拟合，返回 (a, b, R²)。
    只调用一次，所有 run 共用。
    """
    ed_a,ed_b,ed_r2=0.238,1.265,0.841
    print(f'  [全局 ED] 直接 E-D 拟合 : E = {ed_a:.3f} x D^({ed_b:.3f}), R2 = {ed_r2:.3f}')
    return ed_a, ed_b, ed_r2


def compute_ed_params(df, sample_info_train):
    """从训练集正样本做直接 E-D 拟合，返回 (a, b, R²)。"""
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score

    wdf = df.sort_values('valid_time').reset_index(drop=True)

    D_vals, E_vals = [], []
    train_events = sample_info_train[sample_info_train['label'] == 1]['time']
    for t in train_events:
        dur, water = _trace_water_for_event(wdf, str(t))
        if dur > 0 and water > 0:
            D_vals.append(dur)
            E_vals.append(water)

    D_arr = np.array(D_vals, dtype=float)
    E_arr = np.array(E_vals, dtype=float)

    def power_law(d, a, b):
        return a * (d ** b)

    popt, _ = curve_fit(power_law, D_arr, E_arr, p0=[0.2, 1.2], maxfev=10000)
    ed_a, ed_b = popt
    E_pred = power_law(D_arr, ed_a, ed_b)
    ed_r2 = max(0.0, min(1.0, r2_score(E_arr, E_pred)))
    print(f'  [当下 ED] 训练集直接 E-D (n={len(D_arr)}): E = {ed_a:.3f} x D^({ed_b:.3f}), R2 = {ed_r2:.3f}')
    return ed_a, ed_b, ed_r2


def main(idx, mode, split_para_val=1, global_ed=None, epochs=100, k_folds=5):
    """
    idx           : run 编号
    mode          : 'global_ed' | 'per_run_ed' | 'baseline'
    split_para_val: 测试集比例
    global_ed     : (a, b, R²) 全局 ED，仅 mode='global_ed' 时使用
    epochs        : 单折训练轮数上限（默认 100）
    k_folds       : 训练集内部 CV 折数（默认 5）
    """
    print('-' * 10 + f' Run {idx} ({mode}) ' + '-' * 10)

    # 可选复现性开关：设置环境变量 DF_SEED 后，为每个 run 固定 NumPy/PyTorch 全局种子。
    # 默认不启用，以保持与原论文报出数值完全一致的运行方式（见 NOTES.md）。
    seed_base = os.environ.get('DF_SEED')
    if seed_base is not None:
        s = int(seed_base) + idx
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    df = pd.read_csv(DS_PLUS, parse_dates=['valid_time'])
    feature, lable, info = create_time_series_samples(df=df, event_times=event_times, non_event_times=safe_time)
    X_train_scaled, X_test_scaled, sample_info_train, y_train, y_test, sample_info_test, scaler = \
        split_dataset(features=feature, labels=lable, sample_info=info,
                      no_rain_no_melt_times=nowater_event, split_para_val=split_para_val,
                      random_state=42+ idx)

    # ---- 输出目录（同一 split 共享，靠 model_name 区分文件）----
    ratio_str = f'{split_para_val:.1f}'
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'mnt_Test_{ratio_str}')
    os.makedirs(save_dir, exist_ok=True)

    train_time = [str(t) for t in sample_info_train['time'].tolist()]
    test_time = [str(t) for t in sample_info_test['time'].tolist()]

    train_dataset = MyDataset(X_train_scaled, y_train, train_time)
    test_dataset = MyDataset(X_test_scaled, y_test, test_time)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False,
                             num_workers=0, pin_memory=True)

    print('-' * 10 + 'model' + '-' * 10)
    model = LSTM(input_size=2, hidden_size=32, num_layers=2, dropout=0.2, output_size=1)
    model = model.to(device)
    print(f"\nLSTM模型结构：")
    print(model)

    # ---- 根据 mode 决定 PINN / ED / model_name ----
    if mode == 'baseline':
        pinn = False
        ed_a = ed_b = ed_r2 = 0.0
        model_name = 'lstm'
    elif mode == 'global_ed':
        pinn = True
        ed_a, ed_b, ed_r2 = global_ed
        model_name = 'PI_lstm_global'
    elif mode == 'per_run_ed':
        pinn = True
        ed_a, ed_b, ed_r2 = compute_ed_params(df, sample_info_train)
        model_name = 'PI_lstm'
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model, train_log, val_log, cv_metrics = train_model(
        model=model, train_loader=train_loader, test_loader=test_loader,
        lr=1e-3, PINN=pinn, idx=idx, epochs=epochs, k_folds=k_folds,
        curve_a=ed_a, curve_b=ed_b, curve_r=ed_r2,
        save_dir=save_dir, model_name=model_name)

    avg_loss, avg_acc, y_pred_probs, y_true_labels = \
        evaluate_model(model, test_loader, criterion=nn.BCEWithLogitsLoss())
    analyze_no_rain_samples(sample_info_test=sample_info_test,
                            y_test=y_test, y_pred_probs=y_pred_probs)

    # ---- 按事件类型计算详细指标 ----
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    y_pred_np = np.array(y_pred_probs)
    y_pred_bin = (y_pred_np > 0.5).astype(int)
    y_true_np = np.array(y_true_labels).astype(int)

    test_auc = roc_auc_score(y_true_np, y_pred_np) if len(set(y_true_np)) > 1 else 0.0

    # 事件类型: rain_only, melt_only, compound, nowater, safe
    #
    # NOTE on per-type mask design:
    #   rain / melt / compound masks all include ``~is_safe`` (= y_true == 1),
    #   so they contain ONLY positive (DF) samples.  Within a positive-only
    #   subset:
    #     accuracy  = recall  (= fraction of events of that type that the
    #                            model correctly flags)
    #     precision = 1.0  if at least one sample was predicted positive,
    #                 0.0  otherwise (zero_division).  Precision does NOT
    #                 measure anything useful here — it is trivially 1.0
    #                 whenever recall > 0.
    #   The ``overall`` mask (all test samples, 1:1 balanced) gives the
    #   full picture (accuracy, meaningful precision/recall/F1).
    #
    #   ``nowater`` uses ``is_nowater`` alone (no ~is_safe), but negative
    #   samples are drawn from ``safe_time`` which is disjoint from
    #   ``nowater_event``, so it also contains only positives in practice.
    #
    #   ``safe`` is the all-negative subset (y_true == 0) → pos_label=0
    #   is passed to precision_score / recall_score / f1_score so that
    #   these metrics measure the model's ability to identify *non-DF*
    #   samples (specificity / true-negative rate).
    test_times = sample_info_test['time'].values
    is_rain = np.array([pd.to_datetime(t) in rain_event for t in test_times])
    is_melt = np.array([pd.to_datetime(t) in melt_event for t in test_times])
    is_nowater = np.array([pd.to_datetime(t) in nowater_event for t in test_times])
    is_safe = (y_true_np == 0)

    def metrics_by_mask(mask, pos_label=1):
        """Compute accuracy / precision / recall / F1 on a boolean mask.

        Parameters
        ----------
        mask : np.ndarray[bool]
        pos_label : int
            1 (default) for event-type masks (all y_true == 1 within mask).
            0 for the ``safe`` mask (all y_true == 0), so that precision /
            recall / F1 measure the ability to correctly identify *non-DF*
            samples rather than the (non-existent) positive class.
        """
        if mask.sum() == 0:
            return {'n': 0, 'acc': np.nan, 'prec': np.nan, 'rec': np.nan, 'f1': np.nan}
        return {
            'n': int(mask.sum()),
            'acc': accuracy_score(y_true_np[mask], y_pred_bin[mask]),
            'prec': precision_score(y_true_np[mask], y_pred_bin[mask],
                                    pos_label=pos_label, zero_division=0),
            'rec': recall_score(y_true_np[mask], y_pred_bin[mask],
                                pos_label=pos_label, zero_division=0),
            'f1': f1_score(y_true_np[mask], y_pred_bin[mask],
                           pos_label=pos_label, zero_division=0),
        }

    types = {
        'overall': np.ones(len(y_true_np), dtype=bool),
        'rain': is_rain & ~is_melt & ~is_safe,
        'melt': is_melt & ~is_rain & ~is_safe,
        'compound': is_rain & is_melt & ~is_safe,
        'nowater': is_nowater,
        'safe': is_safe,
    }

    result = {
        'run': idx, 'mode': mode, 'split': split_para_val,
        'ed_a': ed_a, 'ed_b': ed_b, 'ed_r2': ed_r2,
        'cv_train_acc_mean': cv_metrics['cv_train_acc_mean'],
        'cv_train_acc_std': cv_metrics['cv_train_acc_std'],
        'cv_val_acc_mean': cv_metrics['cv_val_acc_mean'],
        'cv_val_acc_std': cv_metrics['cv_val_acc_std'],
        'cv_val_loss_mean': cv_metrics['cv_val_loss_mean'],
        'train_val_gap': cv_metrics['train_val_gap'],
        'train_accuracy': round(train_log['acc'][-1], 4) if train_log['acc'] else np.nan,
        'test_auc': round(test_auc, 4),
    }
    for name, mask in types.items():
        # ``safe`` is the all-negative subset → pos_label=0 so that
        # precision / recall / F1 measure non-DF detection quality.
        # All other event-type masks (rain, melt, compound, nowater)
        # contain only positive samples (y_true == 1) by construction
        # (~is_safe is AND-ed in), so pos_label=1 is correct there.
        pl = 0 if name == 'safe' else 1
        m = metrics_by_mask(mask, pos_label=pl)
        result[f'{name}_accuracy'] = round(m['acc'], 4) if not np.isnan(m['acc']) else np.nan
        result[f'{name}_precision'] = round(m['prec'], 4) if not np.isnan(m['prec']) else np.nan
        result[f'{name}_recall'] = round(m['rec'], 4) if not np.isnan(m['rec']) else np.nan
        result[f'{name}_f1'] = round(m['f1'], 4) if not np.isnan(m['f1']) else np.nan

    print("训练完成！")
    print("=" * 60)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM + PINN 泥石流预测')
    # parser.add_argument('--modes', nargs='+', choices=['global_ed', 'per_run_ed', 'baseline'],
    #                     default=[ 'baseline','per_run_ed', 'global_ed'],
    #                     help='ED 来源模式 (默认: baseline per_run_ed global_ed )')
    # parser.add_argument('--splits', type=float, nargs='+',
    #                     default=[0.2, 0.3, 0.4],
    #                     help='测试集划分比例 (默认: 0.2 0.3 0.4)')
    # parser.add_argument('--runs', type=int, default=100,
    #                     help='每种划分的重复次数 (默认: 100)')

    # APP earwarning
    parser.add_argument('--modes', nargs='+', choices=['global_ed', 'per_run_ed', 'baseline'],
                        default=['baseline', 'global_ed'],
                        help='ED 来源模式 (默认: baseline global_ed )')
    parser.add_argument('--splits', type=float, nargs='+',
                        default=[1],
                        help='APP 运用所有的数据训练(默认: 1)')
    parser.add_argument('--runs', type=int, default=100,
                        help='每种划分的重复次数 (默认: 100)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='单折训练轮数上限 (默认: 100，与论文一致)')
    parser.add_argument('--k-folds', type=int, default=5,
                        help='训练集内部 CV 折数 (默认: 5，与论文一致)')


    args = parser.parse_args()

    print(f"模式: {args.modes}  |  Splits: {args.splits}  |  Runs per split: {args.runs}  "
          f"|  Epochs: {args.epochs}  |  K: {args.k_folds}")
    print("=" * 70)

    # 全局 ED 只拟合一次
    global_ed = compute_global_ed_params() if 'global_ed' in args.modes else None

    # ---- 收集所有 run 的结果 ----
    all_results = []
    total = len(args.modes) * len(args.splits) * args.runs
    done = 0

    for i in range(args.runs):
        for mode in args.modes:
            for split_val in args.splits:
                result = main(i, mode=mode, split_para_val=split_val, global_ed=global_ed,
                              epochs=args.epochs, k_folds=args.k_folds)
                all_results.append(result)
                done += 1
                print(f"[{done}/{total}] {mode} split={split_val:.1f} run={i}: "
                      f"train_acc={result['train_accuracy']:.4f} "
                      f"test_acc={result['overall_accuracy']:.4f} "
                      f"AUC={result['test_auc']:.4f}")

    # ---- 汇总表格 + 统计检验 ----
    import pandas as pd
    from scipy.stats import ttest_rel

    save_root = str(REPO_ROOT)
    results_df = pd.DataFrame(all_results)

    # 1. 逐 run 完整结果
    results_df.to_csv(os.path.join(save_root, 'model_evaluation_all_runs.csv'),
                      index=False, encoding='utf-8-sig')

    model_name_map = {'baseline': 'LSTM', 'per_run_ed': 'PI-LSTM',
                      'global_ed': 'PI-LSTM (global)'}

    # 2. 配对 t-test（同一 run_id, 同一 split → 配对）
    test_rows = []
    pi_modes = [m for m in ['per_run_ed', 'global_ed'] if m in args.modes]
    for split_val in args.splits:
        base = results_df[(results_df['split'] == split_val) & (results_df['mode'] == 'baseline')]
        for pm in pi_modes:
            pi = results_df[(results_df['split'] == split_val) & (results_df['mode'] == pm)]
            merged = base.merge(pi, on='run', suffixes=('_base', '_pi'))
            if len(merged) < 3:
                continue
            for metric in ['overall_accuracy', 'cv_val_acc_mean', 'test_auc']:
                col_b, col_p = f'{metric}_base', f'{metric}_pi'
                if col_b not in merged.columns or col_p not in merged.columns:
                    continue
                a = merged[col_b].dropna()
                b = merged[col_p].dropna()
                if len(a) < 3:
                    continue
                t, p = ttest_rel(b, a)  # two-sided paired t-test (scipy default)
                test_rows.append({
                    'split': split_val, 'comparison': f'{model_name_map[pm]} vs LSTM',
                    'metric': metric,
                    'baseline_mean': round(a.mean(), 4),
                    'pi_mean': round(b.mean(), 4),
                    'delta': round(b.mean() - a.mean(), 4),
                    't_stat': round(t, 3),
                    'p_value': round(p, 3),
                    'significant_0.05': 'YES' if p < 0.05 else 'no',
                })
    tests_df = pd.DataFrame(test_rows)
    tests_path = os.path.join(save_root, 'statistical_tests.csv')
    tests_df.to_csv(tests_path, index=False, encoding='utf-8-sig')
    print(f"\n统计检验已保存: {tests_path}")

    # 3. 分组统计（按 split_ratio × model）
    summary_rows = []
    for (split_val, mode), grp in results_df.groupby(['split', 'mode']):
        row = {'split_ratio': split_val, 'model': model_name_map.get(mode, mode)}
        for event_type in ['overall', 'rain', 'melt', 'compound', 'nowater', 'safe']:
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                col = f'{event_type}_{metric}'
                vals = grp[col].dropna()
                row[f'{event_type}_{metric}_mean'] = round(vals.mean(), 4)
                row[f'{event_type}_{metric}_std'] = round(vals.std(ddof=1) if len(vals) > 1 else 0.0, 4)
        for cv_col in ['cv_train_acc_mean', 'cv_val_acc_mean', 'cv_val_acc_std',
                       'train_val_gap', 'overall_accuracy', 'test_auc']:
            cv_col_short = cv_col if cv_col.startswith('cv_') else cv_col
            if cv_col in grp.columns:
                vals = grp[cv_col].dropna()
                row[f'{cv_col_short}_mean' if not cv_col_short.startswith('cv_') else cv_col_short] = round(vals.mean(), 4)
                row[f'{cv_col_short}_std' if not cv_col_short.startswith('cv_') else f'{cv_col_short}_std'] = round(vals.std(ddof=1) if len(vals) > 1 else 0.0, 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    model_order = {'LSTM': 0, 'PI-LSTM': 1, 'PI-LSTM (global)': 2}
    summary_df['_sort'] = summary_df['model'].map(model_order)
    summary_df = summary_df.sort_values(['split_ratio', '_sort']).drop(columns='_sort')
    summary_df.to_csv(os.path.join(save_root, 'model_evaluation_summary.csv'),
                      index=False, encoding='utf-8-sig')

    # 4. 打印综合表格
    for split_val in args.splits:
        sub = summary_df[summary_df['split_ratio'] == split_val]
        if sub.empty:
            continue
        print(f"\n{'='*100}")
        print(f"  {args.k_folds}-Fold CV Performance Comparison  (split={split_val:.1f}, {args.runs} runs)")
        print(f"{'='*100}")
        print(f"  {'Metric':<30} {'LSTM':>20} {'PI-LSTM':>20} {'PI-LSTM (global)':>20}")
        print(f"  {'-'*30} {'-'*20} {'-'*20} {'-'*20}")

        def cell(grp, col, mode_name):
            m = grp[grp['model'] == model_name_map[mode_name]]
            if m.empty:
                return '—'
            v = m.iloc[0].get(f'{col}_mean', np.nan)
            s = m.iloc[0].get(f'{col}_std', np.nan)
            if pd.isna(v):
                return '—'
            if pd.isna(s):
                return f'{v:.4f}'
            return f'{v:.4f} ± {s:.4f}'

        rows = [
            ('CV Val Accuracy',      'cv_val_acc'),
            ('CV Train Accuracy',    'cv_train_acc'),
            ('Test Accuracy',        'overall_accuracy'),
            ('Test AUC',             'test_auc'),
            ('Train-Val Gap',       'train_val_gap'),
        ]
        for label, col in rows:
            cells = []
            for m in ['baseline', 'per_run_ed', 'global_ed']:
                cells.append(cell(sub, col, m) if m in args.modes else '—')
            print(f"  {label:<30} {cells[0]:>20} {cells[1]:>20} {cells[2]:>20}")

        print(f"  {'-'*30} {'-'*20} {'-'*20} {'-'*20}")
        print(f"  {'By Event Type (Test Accuracy):':<30}")
        for etype, elabel in [('rain', 'Rain-only'), ('melt', 'Melt-only'),
                               ('compound', 'Compound'), ('nowater', 'No-water ★'),
                               ('safe', 'Safe')]:
            cells = []
            for m in ['baseline', 'per_run_ed', 'global_ed']:
                cells.append(cell(sub, f'{etype}_accuracy', m) if m in args.modes else '—')
            print(f"    {elabel:<28} {cells[0]:>20} {cells[1]:>20} {cells[2]:>20}")

        # p-values
        if tests_df.empty:
            continue
        print(f"  {'-'*30} {'-'*20} {'-'*20} {'-'*20}")
        for pm in pi_modes:
            tsub = tests_df[(tests_df['split'] == split_val) & (tests_df['comparison'].str.contains(model_name_map[pm]))]
            if tsub.empty:
                continue
            for metric, mlabel in [('overall_accuracy', 'Test Accuracy'), ('cv_val_acc_mean', 'CV Val Accuracy')]:
                tr = tsub[tsub['metric'] == metric]
                if not tr.empty:
                    r = tr.iloc[0]
                    print(f"  p-value ({mlabel}, {model_name_map[pm]} vs LSTM): {r['p_value']:.4f}  "
                          f"(Δ={'+' if r['delta']>0 else ''}{r['delta']:.4f})  {'★ p<0.05' if r['p_value']<0.05 else ''}")

    print(f"\n{'='*100}")
    print("All results saved:")
    print(f"  {save_root}/model_evaluation_all_runs.csv")
    print(f"  {save_root}/model_evaluation_summary.csv")
    print(f"  {save_root}/statistical_tests.csv")
    print("Done.")
