#!/usr/bin/env python3
"""Shared utilities for Assignment 7 — advancing the ENSO staged model plan.

Self-contained (so this folder zips independently). Loads the A5 lag representation
from the A4 locked table, builds train-fit design matrices, and provides the same
metrics/baselines/split used in Assignments 5-6 so results are directly comparable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
LOCKED = ROOT / "data" / "enso_features_monthly.csv"
CLASSES = ["El Nino", "La Nina", "Neutral"]
SEED = 6320

# A5 hand-built lag representation (the current-best portfolio representation).
A5_NUMERIC = [
    "n34_anom", "n34_lag1", "n34_lag2", "n34_lag3", "n34_lag6", "n34_lag12",
    "n34_roll3", "n34_roll5", "n34_trend1", "n34_trend3", "month_sin", "month_cos",
]
CATEGORICAL = ["state_now"]
A5_C = 0.03  # validation-selected L2 inverse strength from Assignment 5


def load_a5() -> pd.DataFrame:
    df = pd.read_csv(LOCKED)
    df = df[df["split"].isin(["train", "val", "test"])].copy()
    df = df[df["has_core_features"] & df["has_target"]]
    df = df.dropna(subset=A5_NUMERIC + CATEGORICAL + ["target_state"])
    df["t"] = pd.to_datetime(df["date"] + "-01")
    df = df.sort_values("t").reset_index(drop=True)
    df["decade"] = (df["year"] // 10 * 10).astype(int).astype(str) + "s"
    df["is_transition"] = df["state_now"] != df["target_state"]
    return df


def splits(df):
    return (df[df.split == "train"].reset_index(drop=True),
            df[df.split == "val"].reset_index(drop=True),
            df[df.split == "test"].reset_index(drop=True))


def design(tr, va, te, numeric=A5_NUMERIC):
    ct = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", categories=[CLASSES],
                              sparse_output=False), CATEGORICAL),
    ])
    Xtr = ct.fit_transform(tr); Xva = ct.transform(va); Xte = ct.transform(te)
    return (ct, Xtr, Xva, Xte, tr.target_state.to_numpy(),
            va.target_state.to_numpy(), te.target_state.to_numpy())


def fit_logreg(Xtr, ytr, C=A5_C, class_weight=None):
    return LogisticRegression(C=C, max_iter=5000, solver="lbfgs",
                              class_weight=class_weight,
                              random_state=SEED).fit(Xtr, ytr)


# ---- metrics ----
def macro_f1(yt, yp): return float(f1_score(yt, yp, labels=CLASSES, average="macro", zero_division=0))
def balanced_acc(yt, yp): return float(balanced_accuracy_score(yt, yp))
def accuracy(yt, yp): return float((np.asarray(yt) == np.asarray(yp)).mean())


def recall_of(yt, yp, cls): return float(recall_score(yt, yp, labels=[cls], average=None, zero_division=0)[0])
def precision_of(yt, yp, cls): return float(precision_score(yt, yp, labels=[cls], average=None, zero_division=0)[0])


def metric_block(yt, yp):
    out = {"accuracy": accuracy(yt, yp), "balanced_accuracy": balanced_acc(yt, yp),
           "macro_f1": macro_f1(yt, yp)}
    for c in CLASSES:
        out[f"recall_{c.replace(' ', '_')}"] = recall_of(yt, yp, c)
    return out


def per_class_table(yt, yp, name):
    rows = []
    for c in CLASSES:
        rows.append({"class": c, "model": name, "support": int((np.asarray(yt) == c).sum()),
                     "recall": recall_of(yt, yp, c), "precision": precision_of(yt, yp, c),
                     "f1": float(f1_score(yt, yp, labels=[c], average=None, zero_division=0)[0])})
    return pd.DataFrame(rows)


def confusion(yt, yp):
    cm = confusion_matrix(yt, yp, labels=CLASSES)
    return pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                        columns=[f"pred_{c}" for c in CLASSES])


# ---- baselines ----
def persistence_pred(frame): return frame["persist_pred_state"].to_numpy()
