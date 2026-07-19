# CS 6320 Assignment 7 — Ryan Poppe

**Portfolio Modeling Progress + Preliminary Transfer-Learning Relevance — NOAA ENSO
+3-month forecasting.**

Portfolio work. Part A runs the exact next experiment named in the Assignment 6
checkpoint: make the A5-lag logistic regression **cost-sensitive** to fix the La Niña
recall weakness, selected on validation per-class recall. Part B is a preliminary,
evidence-based decision that **vision transfer learning is not relevant** to this
numeric climate-index project (with the one nuance where pretrained feature
extraction could enter, and why it is declined).

Memo: `writeups/Ryan-Poppe-assignment-7-writeup.md` (Part A + Part B).

## Layout

```
ut-cs6320-assignment-07/
├── a7_common.py      # load A5-lag representation, split, metrics, baselines (self-contained)
├── train_a7.py       # cost-sensitive class-weight sweep + threshold diagnostic + walk-forward
├── make_plots.py     # 5 plots
├── data/enso_features_monthly.csv     # A4 locked table
├── outputs/          # metrics CSVs, run_summary.json, run_log.txt, plots/
└── writeups/Ryan-Poppe-assignment-7-writeup.md
```

## Run evidence

```bash
cd ut-cs6320-assignment-07
python3 train_a7.py     # -> outputs/model_comparison.csv, classweight_sweep.csv, threshold_sweep.csv,
                        #    per_class_before_after.csv, confusion_before.csv, confusion_after.csv,
                        #    walkforward_lanina.csv, run_summary.json
python3 make_plots.py   # -> outputs/plots/*.png
```

Full console log: `outputs/run_log.txt`.

- **Split:** unchanged locked A4 chronological split (train ≤ 2004 = 648, val
  2005–2014 = 120, test ≥ 2015 = 131). Representation, metrics, and baselines are the
  A5/A6 ones, so results are directly comparable. Selection on **validation**; test
  reserved for final reporting.
- **Seed:** `6320` on every estimator (deterministic lbfgs); re-running reproduces
  `run_summary.json` exactly (verified — no diff).
- **Env:** Python 3, scikit-learn 1.7, numpy, pandas, matplotlib. No network.

## Headline result (test)

| model | La Niña recall | El Niño recall | Neutral recall | macro-F1 | bal-acc |
|---|---|---|---|---|---|
| persistence | 0.513 | 0.625 | 0.404 | 0.516 | 0.514 |
| logreg unweighted (A5) | 0.410 | 0.725 | 0.788 | 0.649 | 0.641 |
| **logreg weighted ×1.5 (A7)** | **0.564** | 0.725 | 0.654 | **0.655** | 0.648 |

The validation-selected cost-sensitive model raises La Niña recall above persistence
**without** losing El Niño recall or overall macro-F1 — the A6 success criterion is
met. The cost is Neutral recall (0.79 → 0.65). Vision transfer learning is **not
relevant** (Part B); the sequence-model question is deferred to Week 9.
