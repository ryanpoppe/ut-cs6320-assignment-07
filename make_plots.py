#!/usr/bin/env python3
"""Assignment 7 plots from outputs/ artifacts. -> outputs/plots/*.png"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import a7_common as C

OUT = C.ROOT / "outputs"; P = OUT / "plots"; P.mkdir(parents=True, exist_ok=True)
GREY = "#8a8a8a"; BLUE = "#2c6fbb"; ORANGE = "#e07b39"; GREEN = "#3a9b6e"; RED = "#c0392b"
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def per_class_recall():
    pc = pd.read_csv(OUT / "per_class_before_after.csv")
    piv = pc.pivot(index="class", columns="model", values="recall").reindex(C.CLASSES)
    x = np.arange(3); w = 0.27
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - w, piv["persistence"], w, color=GREY, label="persistence")
    ax.bar(x, piv["logreg_unweighted"], w, color=ORANGE, label="logreg unweighted (A5)")
    ax.bar(x + w, piv["logreg_weighted"], w, color=BLUE, label="logreg weighted (A7)")
    ax.set_xticks(x); ax.set_xticklabels(C.CLASSES); ax.set_ylabel("test recall"); ax.set_ylim(0, 1.0)
    for i in range(3):
        for off, col in [(-w, "persistence"), (0, "logreg_unweighted"), (w, "logreg_weighted")]:
            ax.text(x[i]+off, piv[col].iloc[i]+0.02, f"{piv[col].iloc[i]:.2f}", ha="center", fontsize=7)
    ax.set_title("A7 cost-sensitive fix: La Nina recall 0.41 -> 0.56 (now > persistence),\n"
                 "El Nino held at 0.73; cost is Neutral recall (0.79 -> 0.65)")
    ax.legend(fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout(); fig.savefig(P / "per_class_recall_before_after.png"); plt.close(fig)


def confusions():
    cb = pd.read_csv(OUT / "confusion_before.csv", index_col=0)
    ca = pd.read_csv(OUT / "confusion_after.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, cm, title in [(axes[0], cb, "BEFORE — unweighted (A5)\n22 La Nina -> Neutral"),
                          (axes[1], ca, "AFTER — weighted x1.5 (A7)\n16 La Nina -> Neutral (6 recovered)")]:
        im = ax.imshow(cm.to_numpy(), cmap="Blues"); mx = cm.to_numpy().max()
        ax.set_xticks(range(3)); ax.set_xticklabels(C.CLASSES, rotation=20)
        ax.set_yticks(range(3)); ax.set_yticklabels(C.CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true (+3mo)")
        for i in range(3):
            for j in range(3):
                v = cm.to_numpy()[i, j]
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > mx/2 else "black", fontsize=12)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Test confusion — moving the La Nina/Neutral boundary", fontsize=12)
    fig.tight_layout(); fig.savefig(P / "confusion_before_after.png"); plt.close(fig)


def classweight_sweep():
    s = pd.read_csv(OUT / "classweight_sweep.csv")
    target = 0.55
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(s.lanina_weight, s.val_recall_LN, "o-", color=BLUE, label="val La Nina recall")
    ax.plot(s.lanina_weight, s.val_recall_Neutral, "s-", color=ORANGE, label="val Neutral recall")
    ax.plot(s.lanina_weight, s.val_bal_acc, "^-", color=GREEN, label="val balanced accuracy")
    ax.axhline(target, color=GREY, ls=":", lw=1.3, label=f"target (val persist LN recall={target})")
    ax.axvline(1.5, color=RED, ls="--", lw=1.4, label="selected weight=1.5")
    ax.set_xlabel("La Nina class weight"); ax.set_ylabel("validation metric"); ax.set_ylim(0.4, 0.95)
    ax.set_title("Cost-sensitive sweep (validation): weight 1.5 is the least\n"
                 "aggressive setting that reaches the La Nina recall target")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout(); fig.savefig(P / "classweight_sweep.png"); plt.close(fig)


def threshold_pr():
    t = pd.read_csv(OUT / "threshold_sweep.csv")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(t.test_recall_LN, t.test_precision_LN, "o-", color=BLUE)
    for _, r in t.iterrows():
        if r.tau_lanina in (1.0, 1.75, 2.75, 3.0):
            ax.annotate(f"x{r.tau_lanina}", (r.test_recall_LN, r.test_precision_LN),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("La Nina recall (test)"); ax.set_ylabel("La Nina precision (test)")
    ax.set_title("Diagnostic: La Nina precision-recall as the decision threshold shifts\n"
                 "(no free lunch — pushing recall up trades precision away)")
    fig.tight_layout(); fig.savefig(P / "threshold_pr.png"); plt.close(fig)


def walkforward():
    d = pd.read_csv(OUT / "walkforward_lanina.csv")
    x = np.arange(len(d)); w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w, d.persist_recall_LN, w, color=GREY, label="persistence")
    ax.bar(x, d.unweighted_recall_LN, w, color=ORANGE, label="unweighted (A5)")
    ax.bar(x + w, d.weighted_recall_LN, w, color=BLUE, label="weighted (A7)")
    ax.set_xticks(x); ax.set_xticklabels(d.fold, rotation=35, fontsize=8)
    ax.set_ylabel("La Nina recall"); ax.set_ylim(0, 1.0)
    ax.set_title("Walk-forward La Nina recall: weighting helps out-of-sample\n"
                 "(mean 0.40 vs unweighted 0.34) but stays modest and noisy")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout(); fig.savefig(P / "walkforward_lanina.png"); plt.close(fig)


def main():
    per_class_recall(); confusions(); classweight_sweep(); threshold_pr(); walkforward()
    print("wrote plots:", sorted(p.name for p in P.glob("*.png")))


if __name__ == "__main__":
    main()
