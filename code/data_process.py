import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import math
import bisect
from datetime import datetime, timedelta
from typing import List, Tuple

from config import INVENTORY, RAIN, HEAT

df = pd.read_csv(INVENTORY, encoding='utf-8-sig')
time_strings = df['发布时间'].astype(str).tolist()
event_times= pd.to_datetime(time_strings)
df = pd.read_csv(RAIN)
df_valid = df[df['duration_hours'] > 0].copy()
rain_event_strings = df_valid['event_time'].astype(str).tolist()
df = pd.read_csv(HEAT)
df_valid = df[df['duration_hours'] > 0].copy()
melt_event_strings = df_valid['event_time'].astype(str).tolist()


# print('现实情况，负样本多，选取2025年的负样本')
# all_time = pd.to_datetime(pd.read_csv(r'F:\paper2\ds_plus.csv')['valid_time'])
# cond_year = (all_time.dt.year == 2025)
# t0 = event_times
# t1 = event_times - pd.Timedelta(hours=1)
# t2 = event_times - pd.Timedelta(hours=2)
# event_dt_hour = pd.concat([
#     t0.dt.strftime('%Y-%m-%d %H'),
#     t1.dt.strftime('%Y-%m-%d %H'),
#     t2.dt.strftime('%Y-%m-%d %H')
# ]).unique()
# all_dt_hour = all_time.dt.strftime('%Y-%m-%d %H')
# cond_not_event=~all_dt_hour.isin(event_dt_hour)
#
# safe_time = all_time[cond_year & cond_not_event]
# print(len(safe_time))
# print('1:1 test  old 修改  不要删除这个原1：1代码')
negative_samples_strings = [
    '2025-07-25 18:54:00',
    '2025-07-24 17:23:00',
    '2025-07-23 08:36:00',
    '2025-07-16 11:24:00',
    '2025-07-15 18:45:00',
    '2025-07-14 23:14:00',
    '2025-07-13 15:59:00',
    '2025-07-12 19:14:00',
    '2025-06-11 22:51:00',
    '2025-05-28 16:29:00',
    '2025-05-27 09:03:00',
    '2024-08-20 21:03:00',
    '2024-08-19 15:42:00',
    '2024-08-01 09:44:00',
    '2024-08-18 02:44:00',
    '2024-08-16 19:01:00',
    '2024-08-12 17:15:00',
    '2024-07-25 17:30:00',
    '2024-08-11 17:39:00',
    '2024-08-07 17:11:00',
    '2024-08-03 20:44:00',
    '2024-07-26 20:33:00',
    '2024-07-25 19:08:00',
    '2024-07-24 04:09:00',
    '2024-07-23 12:19:00',
    '2024-07-22 13:20:00',
    '2024-07-21 16:38:00',
    '2024-07-20 00:00:00',
    '2024-07-19 18:44:00',
    '2022-09-01 13:30:00',
    '2022-08-14 13:48:00',
    '2022-07-30 18:54:00',
    '2022-07-16 18:03:00',
    '2022-07-15 17:07:00',
    '2022-07-14 19:50:00',
    '2022-07-13 18:14:00',
    '2022-07-12 16:07:00',
    '2022-07-01 19:43:00',
    '2022-06-30 12:40:00',
    '2022-06-28 18:22:00',
    '2022-06-27 19:28:00',
    '2022-06-25 09:26:00'
]
safe_time = pd.to_datetime(negative_samples_strings)




rain_event, melt_event = pd.to_datetime(rain_event_strings), pd.to_datetime(melt_event_strings)
mask = ~event_times.isin(rain_event) & ~event_times.isin(melt_event)
nowater_event = event_times[mask]
print(
    f'len(event_times):{len(event_times)}, len(rain_event):{len(rain_event)}, '
    f'len(melt_event):{len(melt_event)}, len(nowater_event):{len(nowater_event)}, '
    f'len(safe_time):{len(safe_time)}')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    df_core = df[['valid_time', 'rainfall_mm', 'temprature_Cel']].copy()
    print(f"Data loaded:")
    print(f"Time range: {df_core['valid_time'].min()} to {df_core['valid_time'].max()}")
    print(f"Records: {len(df_core)} (hourly)")
    return df_core


def create_time_series_samples(df, event_times, non_event_times, time_window=24):
    features = []
    labels = []
    sample_info = []

    for event_time in event_times:
        start_time = event_time - pd.Timedelta(hours=time_window)
        window_data = df[(df['valid_time'] >= start_time) & (df['valid_time'] < event_time)]
        window_data = window_data.sort_values('valid_time')
        feat = window_data[['rainfall_mm', 'temprature_Cel']].values
        if len(feat) < time_window:
            continue
        features.append(feat[:time_window])
        labels.append(1)
        sample_info.append({'time': event_time, 'label': 1})

    for non_event_time in non_event_times:
        start_time = non_event_time - pd.Timedelta(hours=time_window)
        window_data = df[(df['valid_time'] >= start_time) & (df['valid_time'] < non_event_time)]
        window_data = window_data.sort_values('valid_time')
        feat = window_data[['rainfall_mm', 'temprature_Cel']].values
        if len(feat) < time_window:
            continue
        features.append(feat[:time_window])
        labels.append(0)
        sample_info.append({'time': non_event_time, 'label': 0})

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    sample_info = pd.DataFrame(sample_info)

    print(f"\nSamples built:")
    print(f"Total: {len(features)} (debris-flow: {int(sum(labels))}, "
          f"no debris-flow: {len(labels) - int(sum(labels))})")
    print(f"LSTM input shape: {features.shape} (samples, timesteps, features)")
    return features, labels, sample_info


# ===================== 3. Dataset split =====================
def split_dataset(features, labels, sample_info, no_rain_no_melt_times,
                  split_para_val=1, random_state=42):
    """
    Split into train/test. random_state fixes the split so
    different modes with the same run_id get identical train/test sets.
    """
    len_nowater = len(nowater_event)
    len_water = len(event_times) - len_nowater
    len_safe = len(sample_info[(sample_info['label'] == 0)])
    print(f'len_nowater:{len_nowater}, len_water:{len_water}, len_safe:{len_safe}')

    no_water_idx = sample_info[
        (sample_info['label'] == 1) &
        (sample_info['time'].isin(no_rain_no_melt_times))
        ].sample(int(np.ceil(split_para_val * len_nowater)),
                 random_state=random_state).index.tolist()

    normal_debris_idx = sample_info[
        (sample_info['label'] == 1) &
        (~sample_info['time'].isin(no_rain_no_melt_times))
        ].sample(int(np.ceil(split_para_val * len_water)),
                 random_state=random_state + 1).index.tolist()

    non_debris_test_idx = sample_info[
        (sample_info['label'] == 0)
        ].sample(int(np.ceil(split_para_val * len_safe)),
                 random_state=random_state + 2).index.tolist()

    debris_test_idx = no_water_idx + normal_debris_idx
    test_idx = debris_test_idx + non_debris_test_idx
    print(f'test_idx: {test_idx}')
    train_idx = [i for i in range(len(features)) if i not in test_idx]

    if split_para_val == 1:
        train_idx = test_idx

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    sample_info_train = sample_info.iloc[train_idx].copy()
    sample_info_test = sample_info.iloc[test_idx].copy()

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])

    scaler.fit(X_train_2d)
    X_train_scaled = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)

    print(f"\nDataset split:")
    print(f"Train: {len(X_train_scaled)} samples "
          f"(debris-flow: {int(sum(y_train))}, no: {len(y_train) - int(sum(y_train))})")
    print(f"Test:  {len(X_test_scaled)} samples "
          f"(debris-flow: {int(sum(y_test))}, no: {len(y_test) - int(sum(y_test))})")
    print(sample_info_test)

    return X_train_scaled, X_test_scaled, sample_info_train, y_train, y_test, sample_info_test, scaler


import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, X, y, time):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.time = time

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.time[idx]


if __name__ == "__main__":
    print('test dataprocess')
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(nowater_event)
    exit()
    feature, lable, info = create_time_series_samples(
        df=df, event_times=event_times, non_event_times=safe_time)
    X_train_scaled, X_test_scaled, sample_info_train, y_train, y_test, sample_info_test, scaler = \
        split_dataset(features=feature, labels=lable, sample_info=info,
                      no_rain_no_melt_times=nowater_event, split_para_val=0.4,
                      random_state=42)
    train_time = [str(t) for t in sample_info_train['time'].tolist()]
    test_time = [str(t) for t in sample_info_test['time'].tolist()]
    train_dataset = MyDataset(X_train_scaled, y_train, train_time)
    test_dataset = MyDataset(X_test_scaled, y_test, test_time)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=8, shuffle=True,
        num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=8, shuffle=False,
        num_workers=0, pin_memory=True)

    print(X_train_scaled.shape)
    print(X_test_scaled.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(np.array(test_time))
