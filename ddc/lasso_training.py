"""LASSO squared-error selection on human-reviewed binary dancer labels.

Imputation, scaling and alpha selection are fit within video-group folds.
The resulting linear score is not a calibrated probability.
"""

import numpy as np

from .training_features import PREDICTOR_IDS


def matrix(records):
    return np.asarray([[r["values"].get(fid) if r["values"].get(fid) is not None
                        else np.nan for fid in PREDICTOR_IDS] for r in records], float)


def fit_lasso(records):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray([r["label"] for r in records], int)
    groups = np.asarray([r["run"] for r in records])
    if len(y) < 4 or any((y == c).sum() < 2 for c in (0, 1)):
        raise ValueError("LASSO needs at least two reviewed tracks in each class")
    if len(set(groups)) < 2:
        raise ValueError("LASSO needs reviewed tracks from at least two videos")
    x = matrix(records)
    usable = np.flatnonzero(np.sum(np.isfinite(x), axis=0) >= 2)
    if len(usable) < 5:
        raise ValueError("Fewer than five measurable pose features")
    x = x[:, usable]

    def pipeline(alpha):
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), Lasso(alpha=float(alpha), max_iter=20000, tol=1e-5))

    folds = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(x, y, groups))
    candidates = []
    for alpha in np.geomspace(.3, .00001, 24):
        losses = []
        for train, test in folds:
            estimator = pipeline(alpha).fit(x[train], y[train])
            losses.append(float(np.mean((estimator.predict(x[test]) - y[test]) ** 2)))
        fitted = pipeline(alpha).fit(x, y)
        coef = fitted[-1].coef_
        # Do not pad a sparse fit with zero-weight features and call them predictors.
        if np.count_nonzero(np.abs(coef) > 1e-7) >= 5:
            candidates.append((float(np.mean(losses)), float(alpha), fitted))
    if not candidates:
        raise ValueError("LASSO found fewer than five nonzero predictors; collect more varied labels")
    mse, alpha, fitted = min(candidates, key=lambda c: (c[0], -c[1]))
    coef = fitted[-1].coef_
    order = np.argsort(-np.abs(coef), kind="stable")[:5]
    # Refit a five-input model with the selected alpha. Ranking weights remain explicit.
    top_model = pipeline(alpha).fit(x[:, order], y)
    selected = usable[order]
    return {
        "method": "LASSO squared error on dancer=1/non_dancer=0; video-group CV",
        "feature_ids": [PREDICTOR_IDS[i] for i in selected],
        "ranking_coefficients": coef[order].tolist(),
        "coefficient": top_model[-1].coef_.tolist(),
        "intercept": float(top_model[-1].intercept_),
        "medians": top_model[0].statistics_.tolist(),
        "mean": top_model[1].mean_.tolist(), "std": top_model[1].scale_.tolist(),
        "alpha": alpha, "selection_cv_mse": mse,
        "cv_note": "Alpha-selection CV error, not an independent test score",
        "runs": sorted(set(groups.tolist())), "reviewed_tracks": len(records),
    }


def score(row, model):
    values = np.asarray([row["values"].get(fid) if row["values"].get(fid) is not None
                         else np.nan for fid in model["feature_ids"]], float)
    values = np.where(np.isfinite(values), values, model["medians"])
    z = (values - model["mean"]) / model["std"]
    result = float(z @ model["coefficient"] + model["intercept"])
    return {"score": result, "predicted_label": "dancer" if result >= .5 else "non_dancer",
            "contributions": (z * model["coefficient"]).tolist()}
