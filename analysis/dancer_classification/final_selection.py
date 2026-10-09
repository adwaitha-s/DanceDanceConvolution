"""
Final result: embedded feature selection (L1-penalised logistic regression, "LASSO") for
dancer vs. non-dancer, plus the evidence and figures for the write-up.

Headline feature set = the 16 POSE-ONLY features. The 5 detection-quality / camera-position
features (visible_joint_frac, mean_kp_conf, torso_px, x_pos_px, y_pos_px) are extracted and
reported, but excluded from the headline: in this clip spectators happen to sit at the right
edge, farther from the camera, so those features identify the SEAT, not the behaviour.

Outputs (results/ beside this script): ranking.csv, summary.txt, fig1..fig5 png.
    python analysis/dancer_classification/final_selection.py
"""
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import l1_min_c  # noqa: E402

import compare_methods as cm  # noqa: E402
import plotting as fs  # noqa: E402

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
K, OUT = 5, str(HERE / "results")
GROUP = {"wrist_speed_mean": "time", "wrist_jerk_rms": "time", "ankle_speed_mean": "time", "body_sway_std": "time",
         "wrist_dom_freq_hz": "frequency", "beat_band_frac": "frequency", "wrist_spec_entropy": "frequency",
         "stft_centroid_hz": "time-frequency", "stft_flux": "time-frequency",
         "arm_reach": "domain", "wrist_height": "domain", "leg_extension": "domain", "knee_extension": "domain",
         "lr_wrist_asym": "domain", "torso_lean_deg": "domain", "travel_speed": "domain",
         "visible_joint_frac": "human/context", "mean_kp_conf": "human/context", "torso_px": "human/context",
         "x_pos_px": "human/context", "y_pos_px": "human/context"}


def coef_at_k_active(X, y, k=K, n_c=60):
    """Standardised L1-logistic coefficients at the largest C where >= k features are active."""
    Z = StandardScaler().fit_transform(X)
    for C in np.logspace(np.log10(l1_min_c(Z, y, loss="log")), 1.5, n_c):
        m = LogisticRegression(penalty="l1", solver="liblinear", random_state=0, C=C, max_iter=2000).fit(Z, y)
        if (np.abs(m.coef_[0]) > 1e-8).sum() >= k:
            return pd.Series(m.coef_[0], X.columns), C
    return pd.Series(m.coef_[0], X.columns), C


PRUNE = 0.05      # a top-5 feature whose standardised weight is below this adds nothing once the others are in


def eval_pruned(df, feats, folds):
    """Nested score of the final rule: LASSO top-5, then drop features with ~0 weight, then logistic regression.

    Everything (ranking, weights, dropping) is redone inside each training fold. Returns
    (pooled AUC, pooled accuracy, how often each feature survived, mean number kept)."""
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    X, y = df[feats], df.is_dancer
    p = np.zeros(len(df))
    kept_n, kept = [], []
    for tr, te in folds(df):
        top = list(cm.rank_lasso(X[tr], y[tr])[:K])
        w = coef_at_k_active(X[tr], y[tr])[0]
        keep = [f for f in top if abs(w[f]) > PRUNE]
        kept_n.append(len(keep))
        kept += keep
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X.loc[tr, keep], y[tr])
        p[te] = m.predict_proba(X.loc[te, keep])[:, 1]
    return roc_auc_score(y, p), accuracy_score(y, p > 0.5), pd.Series(kept).value_counts(), float(np.mean(kept_n))


def bootstrap_top_k(df, feats, k=K, B=200, seed=0):
    """Cluster bootstrap: resample whole TRACKS (within class) and count how often each feature is top-k."""
    rng = np.random.default_rng(seed)
    by_class = {c: df[df.is_dancer == c].track.unique() for c in (0, 1)}
    cnt = pd.Series(0.0, feats)
    for _ in range(B):
        tr = np.concatenate([rng.choice(v, len(v)) for v in by_class.values()])
        d = pd.concat([df[df.track == t] for t in tr], ignore_index=True)
        cnt[list(cm.rank_lasso(d[feats], d.is_dancer)[:k])] += 1
    return cnt / B


def main():
    fs.style()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(HERE / "features.csv")
    allf = [c for c in df.columns if c not in cm.NON_FEATURES + [cm.TARGET]]
    pose = [c for c in allf if c not in cm.CONTEXT]
    X, y = df[pose], df.is_dancer

    # --- ranking on all data (descriptive), with the evidence behind each feature
    ent = cm.lasso_entry(X, y)
    coef, C5 = coef_at_k_active(X, y)
    stab = bootstrap_top_k(df, pose)
    table = pd.DataFrame({
        "group": [GROUP[f] for f in ent.index], "lasso_rank": range(1, len(ent) + 1),
        "coef_at_5_active": coef[ent.index].round(3), "top5_stability": stab[ent.index].round(2),
        "mean_dancer": df[df.is_dancer == 1][ent.index].mean().round(3),
        "mean_nondancer": df[df.is_dancer == 0][ent.index].mean().round(3),
        "single_feature_AUC": [round(max(roc_auc_score(y, X[f]), 1 - roc_auc_score(y, X[f])), 3) for f in ent.index],
    }, index=ent.index)
    table.to_csv(f"{OUT}/ranking.csv")
    top = list(ent.index[:K])
    final = [f for f in top if abs(coef[f]) > PRUNE]          # drop the redundant one(s)
    pruned = {n: eval_pruned(df, pose, fo) for n, fo in [("unseen time blocks", cm.folds_time),
                                                         ("unseen people", cm.folds_person)]}

    # --- honest scores for the top-K, selection redone inside every training fold
    res = {}
    for cv_name, folds in [("unseen time blocks", cm.folds_time), ("unseen people", cm.folds_person)]:
        r, _ = cm.evaluate(df, pose, K, folds, methods=["Embedded: LASSO (L1 logistic)"], baseline=True)
        res[cv_name] = r
    res_all = {}
    for cv_name, folds in [("unseen time blocks", cm.folds_time), ("unseen people", cm.folds_person)]:
        res_all[cv_name] = cm.evaluate(df, pose, K, folds)[0]

    # --- figures
    corr_cols = list(ent.index[:10])
    fs.corr_heatmap(X, corr_cols, f"{OUT}/fig1_correlation.png")

    c = coef[ent.index[:8]][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 0.36 * len(c) + 1.0))
    ax.barh(c.index, c.values, color=[fs.ORANGE if f in final else fs.GRAY for f in c.index], height=0.7)
    for i, v in enumerate(c.values):
        ax.text(v, i, f" {v:+.2f}" if v else " 0", va="center", ha="left" if v >= 0 else "right", fontsize=9, color=fs.INK2)
    ax.axvline(0, color=fs.INK2, lw=0.8)
    ax.set_title(f"Embedded: LASSO coefficients ({len(final)} selected in orange)")
    ax.set_xlabel("Log-odds of 'dancer' per +1 SD (grey = not selected; ~0 = redundant with a feature already in)")
    ax.grid(axis="y", visible=False)
    fig.savefig(f"{OUT}/fig2_lasso_coef.png", facecolor="white")
    plt.close(fig)

    fs.bar_rank(stab.sort_values(ascending=False), K, "Stability: how often a feature makes the top 5",
                "Share of 200 person-level bootstrap resamples", f"{OUT}/fig3_stability.png", n=10)

    fig, axes = plt.subplots(1, len(final), figsize=(2.3 * len(final), 3.4))
    for ax, f in zip(axes, final):
        data = [df[df.is_dancer == 0][f], df[df.is_dancer == 1][f]]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, medianprops=dict(color="white", lw=1.8), showfliers=False)
        for patch, col in zip(bp["boxes"], [fs.GRAY, fs.ORANGE]):
            patch.set(facecolor=col, edgecolor="none")
        ax.set_xticks([1, 2], ["spectator", "dancer"], fontsize=9)
        ax.set_title(f.replace("_", " "), fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Selected {len(final)} features by class", x=0.01, y=1.04, ha="left", fontweight="bold")
    fig.savefig(f"{OUT}/fig4_selected_by_class.png", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, (cv_name, r) in zip(axes, res_all.items()):
        r = r["logistic AUC"]
        cols = [fs.ORANGE if "LASSO" in m else (fs.GRAY if m == "All features" else fs.NAVY) for m in r.index]
        ax.barh(r.index[::-1], r.values[::-1], color=cols[::-1], height=0.65)
        for i, v in enumerate(r.values[::-1]):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=9, color=fs.INK2)
        ax.set_xlim(0.9, 1.0)
        ax.set_title(f"Test on: {cv_name}", fontsize=11)
        ax.set_xlabel("ROC AUC (top-5, logistic)")
        ax.grid(axis="y", visible=False)
    fig.savefig(f"{OUT}/fig5_method_comparison.png", facecolor="white")
    plt.close(fig)

    # --- summary
    lines = [f"Dancer vs non-dancer | {len(df)} windows ({int(y.sum())} dancer / {int((1 - y).sum())} spectator), "
             f"{df[df.is_dancer == 1].track.nunique()} dancers + {df[df.is_dancer == 0].track.nunique()} spectator tracks",
             f"Method: embedded, L1-logistic (LASSO), features ranked by order of entry into the model",
             f"Feature set: {len(pose)} pose-only features", "", f"Top {K} features:"]
    for f in top:
        r = table.loc[f]
        lines.append(f"  {f:18s} [{r.group}] coef {r.coef_at_5_active:+.2f}  stability {r.top5_stability:.2f}  "
                     f"single-feature AUC {r.single_feature_AUC:.3f}  (dancer {r.mean_dancer} vs spectator {r.mean_nondancer})")
    lines += ["", f"Final set after dropping near-zero weights (|w| < {PRUNE}): {final}",
              "Nested score of that rule (top-5, drop ~0 weights, logistic; all redone inside each training fold):"]
    for n, (auc, acc, kept, mean_k) in pruned.items():
        lines.append(f"  {n:22s} AUC {auc:.3f}  acc {acc:.3f}  mean #features kept {mean_k:.1f}  "
                     f"survival: " + ", ".join(f"{f} {c}" for f, c in kept.items()))
    lines += ["", "Out-of-fold score of the top-5 (selection redone inside each training fold):"]
    for cv_name, r in res.items():
        a = r.loc["Embedded: LASSO (L1 logistic)"]
        b = r.loc["All features"]
        lines.append(f"  {cv_name:22s} logistic: acc {a['logistic acc']:.3f}  AUC {a['logistic AUC']:.3f}   "
                     f"random forest: acc {a['random forest acc']:.3f}  AUC {a['random forest AUC']:.3f}   "
                     f"| all {len(pose)} features, logistic: acc {b['logistic acc']:.3f}  AUC {b['logistic AUC']:.3f}")
    hi = X[corr_cols].corr().abs().where(~np.eye(len(corr_cols), dtype=bool)).stack()
    hi = hi[hi > 0.8]
    seen = set()
    lines += ["", "Highly correlated pairs (|r| > 0.8) among the top-10:"]
    for (a_, b_), r_ in hi.sort_values(ascending=False).items():
        if (b_, a_) not in seen:
            seen.add((a_, b_))
            lines.append(f"  {a_} ~ {b_}: r = {X[a_].corr(X[b_]):.2f}")
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nFull table:\n", table.to_string())


if __name__ == "__main__":
    main()
