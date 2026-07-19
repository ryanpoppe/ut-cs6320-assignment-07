#!/usr/bin/env python3
"""
Assignment 7 — advance the staged plan: fix the La Nina recall weakness.

A6's next-experiment was: make the A5-lag multinomial logistic regression
cost-sensitive (or threshold-tuned) so La Nina recall reaches at least
persistence's level, WITHOUT giving back the El Nino/Neutral gains. One focused
experiment (cost-sensitive learning) + one diagnostic variant (decision threshold).

Selection is on VALIDATION only (test reserved for final reporting). Rule:
among candidates whose val La Nina recall >= validation-persistence La Nina recall,
pick the highest val balanced accuracy (tie-break: least aggressive setting);
if none reach the target, pick the highest val La Nina recall.

Outputs (outputs/): model_comparison.csv, classweight_sweep.csv, threshold_sweep.csv,
per_class_before_after.csv, confusion_before.csv, confusion_after.csv,
walkforward_lanina.csv, run_summary.json. Seed 6320.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import a7_common as C

warnings.filterwarnings("ignore")
OUT = C.ROOT / "outputs"; OUT.mkdir(exist_ok=True)
LN = "La Nina"
WEIGHT_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
TAU_GRID = [round(x, 2) for x in np.arange(1.0, 3.01, 0.25)]


def add(rows, model, settings, val_mm, test_mm, note):
    r = {"model": model, "settings": settings}
    r["val_macro_f1"] = round(val_mm["macro_f1"], 3)
    r["val_recall_La_Nina"] = round(val_mm["recall_La_Nina"], 3)
    r.update({f"test_{k}": round(v, 3) for k, v in test_mm.items()})
    r["note"] = note
    rows.append(r)


def threshold_predict(proba, classes, tau_lanina):
    """argmax after multiplying the La Nina posterior by tau (decision-threshold shift)."""
    P = proba.copy()
    j = list(classes).index(LN)
    P[:, j] *= tau_lanina
    return np.array(classes)[P.argmax(axis=1)]


def select(cands, target_recall):
    """cands: list of dicts with val_recall_LN, val_bal_acc, 'key'. Apply the rule."""
    ok = [c for c in cands if c["val_recall_LN"] >= target_recall - 1e-9]
    pool = ok if ok else cands
    key = "val_bal_acc" if ok else "val_recall_LN"
    return max(pool, key=lambda c: (round(c[key], 6), -c["order"]))


def main():
    df = C.load_a5(); tr, va, te = C.splits(df)
    ct, Xtr, Xva, Xte, ytr, yva, yte = C.design(tr, va, te)
    classes = list(C.CLASSES)
    summary = {"seed": C.SEED, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    rows = []

    # validation-persistence La Nina recall = the selection target
    target = C.recall_of(yva, C.persistence_pred(va), LN)
    summary["val_persistence_recall_LaNina"] = round(target, 3)
    summary["test_persistence_recall_LaNina"] = round(C.recall_of(yte, C.persistence_pred(te), LN), 3)

    # --- reference rows: persistence + A5 unweighted logreg ---
    add(rows, "persistence", "domain rule",
        C.metric_block(yva, C.persistence_pred(va)), C.metric_block(yte, C.persistence_pred(te)),
        "prior bar; La Nina recall to match")
    base = C.fit_logreg(Xtr, ytr)  # unweighted A5 champion
    add(rows, "logreg_unweighted", f"A5 champion, C={C.A5_C}, class_weight=None",
        C.metric_block(yva, base.predict(Xva)), C.metric_block(yte, base.predict(Xte)),
        "A5/A6 result to beat on La Nina")
    conf_before = C.confusion(yte, base.predict(Xte)); conf_before.to_csv(OUT / "confusion_before.csv")

    # --- PRIMARY: cost-sensitive class-weight sweep ---
    sweep = []
    for i, w in enumerate(WEIGHT_GRID):
        cw = {"El Nino": 1.0, "La Nina": w, "Neutral": 1.0}
        m = C.fit_logreg(Xtr, ytr, class_weight=cw)
        vb = C.metric_block(yva, m.predict(Xva))
        sweep.append({"lanina_weight": w, "order": i, "val_recall_LN": vb["recall_La_Nina"],
                      "val_bal_acc": vb["balanced_accuracy"], "val_macro_f1": vb["macro_f1"],
                      "val_recall_El_Nino": vb["recall_El_Nino"], "val_recall_Neutral": vb["recall_Neutral"],
                      "_model": m})
    pd.DataFrame([{k: v for k, v in s.items() if k != "_model"} for s in sweep]).to_csv(
        OUT / "classweight_sweep.csv", index=False)
    chosen = select(sweep, target)
    wsel = chosen["lanina_weight"]; msel = chosen["_model"]
    summary["selected_lanina_weight"] = wsel
    add(rows, "logreg_weighted", f"class_weight La Nina={wsel} (val-selected)",
        C.metric_block(yva, msel.predict(Xva)), C.metric_block(yte, msel.predict(Xte)),
        "PRIMARY cost-sensitive fix")
    conf_after = C.confusion(yte, msel.predict(Xte)); conf_after.to_csv(OUT / "confusion_after.csv")

    # also report the standard 'balanced' setting for reference
    mbal = C.fit_logreg(Xtr, ytr, class_weight="balanced")
    add(rows, "logreg_balanced", "class_weight='balanced'",
        C.metric_block(yva, mbal.predict(Xva)), C.metric_block(yte, mbal.predict(Xte)),
        "standard inverse-frequency weighting")

    # --- DIAGNOSTIC: decision-threshold (La Nina posterior boost) on the unweighted model ---
    proba_va = base.predict_proba(Xva); proba_te = base.predict_proba(Xte)
    tsweep = []
    for i, tau in enumerate(TAU_GRID):
        yv = threshold_predict(proba_va, classes, tau)
        tsweep.append({"tau_lanina": tau, "order": i,
                       "val_recall_LN": C.recall_of(yva, yv, LN),
                       "val_precision_LN": C.precision_of(yva, yv, LN),
                       "val_bal_acc": C.balanced_acc(yva, yv), "val_macro_f1": C.macro_f1(yva, yv),
                       "test_recall_LN": C.recall_of(yte, threshold_predict(proba_te, classes, tau), LN),
                       "test_precision_LN": C.precision_of(yte, threshold_predict(proba_te, classes, tau), LN)})
    pd.DataFrame(tsweep).to_csv(OUT / "threshold_sweep.csv", index=False)
    tchosen = select([{**t, "val_recall_LN": t["val_recall_LN"]} for t in tsweep], target)
    tau_sel = tchosen["tau_lanina"]
    yv_t = threshold_predict(proba_va, classes, tau_sel); yt_t = threshold_predict(proba_te, classes, tau_sel)
    summary["selected_tau_lanina"] = tau_sel
    add(rows, "logreg_threshold", f"La Nina posterior x{tau_sel} (val-selected)",
        C.metric_block(yva, yv_t), C.metric_block(yte, yt_t), "DIAGNOSTIC threshold variant")

    comp = pd.DataFrame(rows); comp.to_csv(OUT / "model_comparison.csv", index=False)

    # --- per-class before/after (test) ---
    pc = pd.concat([
        C.per_class_table(yte, base.predict(Xte), "logreg_unweighted"),
        C.per_class_table(yte, msel.predict(Xte), "logreg_weighted"),
        C.per_class_table(yte, C.persistence_pred(te), "persistence"),
    ], ignore_index=True)
    pc.round(3).to_csv(OUT / "per_class_before_after.csv", index=False)

    # --- walk-forward: La Nina recall + macro-F1, unweighted vs weighted vs persistence ---
    wf = []
    blocks = [(y, y + 1) for y in range(2005, 2025, 2)] + [(2025, 2025)]
    for y0, y1 in blocks:
        trn = df[df.year < y0]; tst = df[(df.year >= y0) & (df.year <= y1)]
        if len(tst) < 8 or trn.target_state.nunique() < 3:
            continue
        _, wXtr, _, wXte, wytr, _, wyte = C.design(trn, trn, tst)
        mu = C.fit_logreg(wXtr, wytr)
        mw = C.fit_logreg(wXtr, wytr, class_weight={"El Nino": 1.0, "La Nina": wsel, "Neutral": 1.0})
        wf.append({"fold": f"{y0}-{y1}", "n": len(tst),
                   "persist_recall_LN": C.recall_of(wyte, C.persistence_pred(tst), LN),
                   "unweighted_recall_LN": C.recall_of(wyte, mu.predict(wXte), LN),
                   "weighted_recall_LN": C.recall_of(wyte, mw.predict(wXte), LN),
                   "unweighted_macro_f1": C.macro_f1(wyte, mu.predict(wXte)),
                   "weighted_macro_f1": C.macro_f1(wyte, mw.predict(wXte))})
    wfd = pd.DataFrame(wf); wfd.to_csv(OUT / "walkforward_lanina.csv", index=False)
    summary["walkforward"] = {
        "unweighted_mean_recall_LN": round(float(wfd.unweighted_recall_LN.mean()), 3),
        "weighted_mean_recall_LN": round(float(wfd.weighted_recall_LN.mean()), 3),
        "persist_mean_recall_LN": round(float(wfd.persist_recall_LN.mean()), 3),
        "unweighted_mean_macro_f1": round(float(wfd.unweighted_macro_f1.mean()), 3),
        "weighted_mean_macro_f1": round(float(wfd.weighted_macro_f1.mean()), 3),
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 220)
    print("=== SELECTION TARGET: val persistence La Nina recall =", round(target, 3), "===")
    print("\n=== MODEL COMPARISON (val selection | test final) ===")
    print(comp[["model", "settings", "val_recall_La_Nina", "test_macro_f1",
                "test_balanced_accuracy", "test_recall_El_Nino", "test_recall_La_Nina",
                "test_recall_Neutral"]].to_string(index=False))
    print("\n=== CLASS-WEIGHT SWEEP (val) ===")
    print(pd.DataFrame([{k: round(v, 3) if isinstance(v, float) else v
                         for k, v in s.items() if k not in ("_model", "order")} for s in sweep]).to_string(index=False))
    print(f"\nSelected La Nina weight = {wsel} ; selected threshold tau = {tau_sel}")
    print("\n=== CONFUSION before (unweighted) ==="); print(conf_before.to_string())
    print("\n=== CONFUSION after (weighted) ===");    print(conf_after.to_string())
    print("\n=== WALK-FORWARD La Nina recall ==="); print(wfd.round(3).to_string(index=False))
    print("\n=== SUMMARY ==="); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
