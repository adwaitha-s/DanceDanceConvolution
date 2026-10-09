"""
Which feature-selection method works best for dancer vs. non-dancer on THIS data?

Plain shuffled k-fold CV leaks here: our windows overlap by 67% and a person's
windows are near-copies of each other, so shuffled folds leak. Two stricter tests here:

  time   leave-one-6s-block-out: all people in a time block are held out together, and
         same-track training windows within one window length of the block are purged.
  person leave-people-out (3 folds): each fold holds out one dancer and ~3 spectator tracks
         entirely -- "does the model recognise a dancer it has never seen?"

Features are ranked INSIDE each training fold (nested), so selection never sees test data.
Each method picks its top-k; a logistic regression (linear) and a random forest (non-linear)
are then scored on the held-out part. Reported: pooled out-of-fold accuracy and ROC AUC.

    python analysis/dancer_classification/compare_methods.py [--k 5]
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
TARGET, NON_FEATURES, BLOCK_S, WIN_S = "is_dancer", ["track", "t_start"], 6.0, 1.5
CONTEXT = ["visible_joint_frac", "mean_kp_conf", "torso_px", "x_pos_px", "y_pos_px"]


def rank_mi(X, y):
    return pd.Series(mutual_info_classif(X, y, random_state=0), X.columns).sort_values(ascending=False).index


def rank_rf(X, y):
    rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X, y)
    return pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False).index


def lasso_entry(X, y, n_c=40):
    """Rank features by how early they enter an L1-penalised logistic regression.

    Walks the penalty from 'everything zero' downward and records the first strength at
    which each coefficient becomes non-zero (earlier = stronger, more independent evidence).
    No single C has to be chosen, and there are no ties among zero coefficients."""
    Z = StandardScaler().fit_transform(X)
    Cs = np.logspace(np.log10(l1_min_c(Z, y, loss="log")), 1.5, n_c)
    entry = np.full(Z.shape[1], np.inf)
    clf = LogisticRegression(penalty="l1", solver="liblinear", random_state=0, max_iter=2000)
    for i, C in enumerate(Cs):
        nz = np.abs(clf.set_params(C=C).fit(Z, y).coef_[0]) > 1e-8
        entry[nz & np.isinf(entry)] = i
    last = np.abs(clf.coef_[0])
    return pd.DataFrame({"entry": entry, "last": last}, X.columns).sort_values(["entry", "last"], ascending=[True, False])


def rank_lasso(X, y):
    return lasso_entry(X, y).index


def rank_rfe(X, y):
    r = RFE(LogisticRegression(max_iter=3000), n_features_to_select=1).fit(StandardScaler().fit_transform(X), y)
    return pd.Series(r.ranking_, X.columns).sort_values().index


METHODS = {"Filter: mutual information": rank_mi, "Embedded: random forest": rank_rf,
           "Embedded: LASSO (L1 logistic)": rank_lasso, "Wrapper: RFE (logistic)": rank_rfe}
SCORERS = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
           "random forest": lambda: RandomForestClassifier(300, random_state=0, n_jobs=-1)}


def folds_time(df):
    block = (df.t_start // BLOCK_S).astype(int)
    for b in sorted(block.unique()):
        test = (block == b).values
        near = np.zeros(len(df), bool)
        for trk, g in df[test].groupby("track"):
            same = (df.track == trk).values
            near |= same & np.array([np.abs(g.t_start.values - t).min() < WIN_S for t in df.t_start])
        yield ~test & ~near, test


def folds_person(df):
    dancers = sorted(df[df[TARGET] == 1].track.unique())
    spect = df[df[TARGET] == 0].groupby("track").size().sort_values(ascending=False).index.tolist()
    groups = {d: [d] for d in dancers}
    for i, s in enumerate(spect):                      # deal spectator tracks round-robin, largest first
        groups[dancers[i % len(dancers)]].append(s)
    for tracks in groups.values():
        test = df.track.isin(tracks).values
        yield ~test, test


def evaluate(df, feats, k, folds, methods=None, baseline=True):
    X, y = df[feats], df[TARGET]
    methods = methods or list(METHODS)
    names = methods + (["All features"] if baseline else [])
    proba = {(m, s): np.zeros(len(df)) for m in names for s in SCORERS}
    picks = {m: [] for m in METHODS}
    for tr, te in folds(df):
        for m in names:
            top = list(METHODS[m](X[tr], y[tr])[:k]) if m in METHODS else list(feats)
            picks.setdefault(m, []).append(top)
            for s, mk in SCORERS.items():
                proba[(m, s)][te] = mk().fit(X.loc[tr, top], y[tr]).predict_proba(X.loc[te, top])[:, 1]
    res = {m: {} for m in names}
    for (m, s), p in proba.items():
        res[m][f"{s} acc"] = accuracy_score(y, p > 0.5)
        res[m][f"{s} AUC"] = roc_auc_score(y, p)
    return pd.DataFrame(res).T, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).with_name("features.csv")))
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    all_feats = [c for c in df.columns if c not in NON_FEATURES + [TARGET]]
    sets = {f"all {len(all_feats)} features": all_feats,
            f"pose-only ({len(all_feats) - len(CONTEXT)}; no detection-quality / camera-position features)":
                [c for c in all_feats if c not in CONTEXT]}
    for fs_name, feats in sets.items():
        for cv_name, folds in [("time blocks", folds_time), ("unseen people", folds_person)]:
            res, picks = evaluate(df, feats, a.k, folds)
            print(f"\n=== {fs_name} | held out: {cv_name} | top-{a.k}, nested ===")
            print(res.round(3).to_string())
            print("most frequently picked:")
            for m, lists in picks.items():
                if m == "All features":
                    continue
                cnt = pd.Series([f for l in lists for f in l]).value_counts().head(a.k)
                print(f"  {m:30s} " + ", ".join(f"{f} ({n}/{len(lists)})" for f, n in cnt.items()))


if __name__ == "__main__":
    main()
