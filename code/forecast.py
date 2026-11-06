#!/usr/bin/env python3
"""Hourly ensemble inference for the 2026 early-warning experiment.

Reproduces the hourly ensemble scores reported for 1 June - 31 August 2026
(Table 3 and Fig. 5 of the manuscript).

The ensemble contains N_PI knowledge-informed models and N_LSTM baseline
models.  Each model casts a binary vote (p > 0.5); the ensemble score is the
percentage of models voting "debris flow".  The E-D score is reported
alongside for reference.

Usage
-----
    python forecast.py --models ../models
    python forecast.py --models ../models --start 2026-06-01 --end 2026-08-31

The trained weights are not distributed with this repository (see README);
train them first with ``main.py --modes baseline global_ed --splits 1 --runs 100``.
"""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import DS_2026, RESULT_DIR
from model import LSTM

# Physical constants used for the E-D reference score (unchanged from the runs
# reported in the manuscript; see NOTES.md).
DIF = (5100 - 3641) / 100 * (-0.71)
DDF = 3.44
DRY_THRESHOLD = 0.1
ED_ALPHA, ED_BETA = 0.238, 1.265

HIDDEN, LAYERS, DROPOUT = 32, 2, 0.2
WINDOW = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(models_dir, n_pi, n_lstm):
    """Load the PI (knowledge-informed) and baseline ensemble members."""
    pi_models, lstm_models = [], []
    for name_list, prefix in ((pi_models, "PI_lstm_global"), (lstm_models, "lstm")):
        for i in range(n_pi if prefix.startswith("PI") else n_lstm):
            path = Path(models_dir) / f"{prefix}{i}.pth"
            if not path.exists():
                raise FileNotFoundError(
                    f"missing ensemble member: {path}\n"
                    "Train the models first (see README) or pass --n-pi/--n-lstm."
                )
            model = LSTM(2, HIDDEN, LAYERS, DROPOUT, 1)
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            model.to(DEVICE).eval()
            name_list.append(model)
    return pi_models, lstm_models


def build_windows(df, window=WINDOW, stride=1):
    """Sliding windows of `window` hours; the label time is the window end."""
    feats = df[["rainfall_mm", "temprature_Cel"]].to_numpy(dtype=np.float32)
    times = df["valid_time"].to_numpy()
    X, t = [], []
    for i in range(window, len(df), stride):
        X.append(feats[i - window:i])
        t.append(times[i])
    return np.asarray(X), np.asarray(t)


@torch.no_grad()
def ensemble_scores(models, X, weight_per_model):
    all_probs = np.zeros((len(models), len(X)), dtype=np.float32)
    for i, model in enumerate(models):
        for start in range(0, len(X), 4096):
            chunk = torch.from_numpy(X[start:start + 4096]).to(DEVICE)
            all_probs[i, start:start + 4096] = torch.sigmoid(model(chunk)).cpu().numpy().ravel()
    votes = (all_probs > 0.5).astype(np.float32)
    return votes.sum(axis=0) * weight_per_model, all_probs.mean(axis=0)


def water_before_events(df, times, max_lookback=24):
    """Cumulated water equivalent and duration of the wet spell before each time."""
    wet = df["rainfall_mm"].where(df["rainfall_mm"] > DRY_THRESHOLD, 0.0).to_numpy(dtype=float) + \
        np.maximum(df["temprature_Cel"].to_numpy(dtype=float) + DIF, 0.0) * (DDF / 24.0)
    idx = pd.Series(np.arange(len(df)), index=df["valid_time"])
    durs, waters = [], []
    for t in times:
        end = idx.get(pd.Timestamp(t))
        if end is None:
            durs.append(0); waters.append(0.0); continue
        dur, water = 0, 0.0
        for k in range(end - 1, max(end - 1 - max_lookback, -1), -1):
            if wet[k] <= 0:
                break
            water += wet[k]
            dur += 1
        durs.append(dur); waters.append(round(water, 2))
    return np.asarray(durs), np.asarray(waters)


def ed_score(durs, waters):
    out = np.zeros(len(durs), dtype=float)
    mask = durs > 0
    out[mask] = np.clip(waters[mask] / (ED_ALPHA * durs[mask] ** ED_BETA), 0.0, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DS_2026))
    ap.add_argument("--models", default="models")
    ap.add_argument("--n-pi", type=int, default=100)
    ap.add_argument("--n-lstm", type=int, default=100)
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-08-31")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=str(Path(RESULT_DIR) / "forecast_2026_hourly.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["valid_time"]).sort_values("valid_time")
    df = df.reset_index(drop=True)
    # The exported ERA5 timestamps are UTC+8 (a day starts at 08:00), so the end
    # of the requested period falls 8 h after midnight of the following day.
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end) + timedelta(days=1, hours=8)
    df = df[(df["valid_time"] >= start - timedelta(hours=WINDOW)) &
            (df["valid_time"] < end)].reset_index(drop=True)
    print(f"ERA5 rows: {len(df)} ({df['valid_time'].min()} - {df['valid_time'].max()})")

    X, times = build_windows(df, stride=args.stride)
    times = pd.to_datetime(times)
    keep = (times >= start) & (times < end)
    X, times = X[keep], times[keep]
    print(f"windows: {len(X)}")

    pi_models, lstm_models = load_models(args.models, args.n_pi, args.n_lstm)
    pi_score, pi_prob = ensemble_scores(pi_models, X, 100.0 / args.n_pi)
    lstm_score, lstm_prob = ensemble_scores(lstm_models, X, 100.0 / args.n_lstm)

    durs, waters = water_before_events(df, times)
    out = pd.DataFrame({
        "valid_time": times,
        "pi_ensemble_score": np.round(pi_score, 2),
        "pi_mean_prob": np.round(pi_prob, 4),
        "lstm_ensemble_score": np.round(lstm_score, 2),
        "lstm_mean_prob": np.round(lstm_prob, 4),
        "duration_h": durs,
        "water_mm": waters,
        "ed_score": np.round(ed_score(durs, waters), 4),
        "pi_alarm": (pi_score >= 50).astype(int),
        "lstm_alarm": (lstm_score >= 50).astype(int),
    })
    os.makedirs(Path(args.out).parent, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"alarms (PI): {int(out['pi_alarm'].sum())} / {len(out)} hours -> {args.out}")


if __name__ == "__main__":
    main()
