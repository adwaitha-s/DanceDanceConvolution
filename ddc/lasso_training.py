"""LASSO squared-error selection on human-reviewed binary dancer labels.

Imputation, scaling and alpha selection are fit within video-group folds.
The resulting linear score is not a calibrated probability.
"""

import numpy as np

from .training_features import PREDICTOR_IDS


def matrix(records):
    return np.asarray([[r["values"].get(fid) if r["values"].get(fid) is not None
                        else np.nan for fid in PREDICTOR_IDS] for r in records], float)


def _prepare(x):
    """Fit median imputation and standardization on this training fold only."""
    medians = np.array([np.median(column[np.isfinite(column)])
                        if np.isfinite(column).any() else 0.
                        for column in x.T], float)
    filled = np.where(np.isfinite(x), x, medians)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    std = np.where(std > 1e-12, std, 1.)
    return (filled - mean) / std, medians, mean, std


def _lasso(x, y, alpha):
    """Coordinate descent for ||y - Xw - b||²/(2n) + alpha*||w||₁."""
    centered_y = y - y.mean()
    gram = x.T @ x / len(x)
    residual_covariance = x.T @ centered_y / len(x)
    coef = np.zeros(x.shape[1], float)
    for _ in range(3000):
        largest_change = 0.
        for j in range(len(coef)):
            variance = gram[j, j]
            if variance <= 1e-12:
                continue
            unpenalized = residual_covariance[j] + variance * coef[j]
            updated = np.sign(unpenalized) * max(abs(unpenalized) - alpha, 0.) / variance
            change = updated - coef[j]
            if change:
                coef[j] = updated
                residual_covariance -= gram[:, j] * change
                largest_change = max(largest_change, abs(change))
        if largest_change < 1e-7:
            break
    return coef, float(y.mean())


def _folds(groups):
    """Balance whole videos among up to five folds; never split one video's tracks."""
    names, counts = np.unique(groups, return_counts=True)
    fold_groups = [[] for _ in range(min(5, len(names)))]
    sizes = [0] * len(fold_groups)
    for name, count in sorted(zip(names, counts), key=lambda item: (-item[1], item[0])):
        fold = int(np.argmin(sizes))
        fold_groups[fold].append(name)
        sizes[fold] += int(count)
    for held_out in fold_groups:
        test = np.flatnonzero(np.isin(groups, held_out))
        train = np.flatnonzero(~np.isin(groups, held_out))
        yield train, test


def fit_lasso(records):
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

    folds = list(_folds(groups))
    candidates = []
    for alpha in np.geomspace(.3, .00001, 24):
        losses = []
        for train, test in folds:
            z, medians, mean, std = _prepare(x[train])
            fold_coef, fold_intercept = _lasso(z, y[train], alpha)
            held_out = (np.where(np.isfinite(x[test]), x[test], medians) - mean) / std
            losses.append(float(np.mean((held_out @ fold_coef + fold_intercept - y[test]) ** 2)))
        z, medians, mean, std = _prepare(x)
        coef, _ = _lasso(z, y, alpha)
        # Do not pad a sparse fit with zero-weight features and call them predictors.
        if np.count_nonzero(np.abs(coef) > 1e-7) >= 5:
            candidates.append((float(np.mean(losses)), float(alpha), coef))
    if not candidates:
        raise ValueError("LASSO found fewer than five nonzero predictors; collect more varied labels")
    mse, alpha, coef = min(candidates, key=lambda c: (c[0], -c[1]))
    order = np.argsort(-np.abs(coef), kind="stable")[:5]
    # Refit a five-input model with the selected alpha. Ranking weights remain explicit.
    top_z, top_medians, top_mean, top_std = _prepare(x[:, order])
    top_coef, top_intercept = _lasso(top_z, y, alpha)
    selected = usable[order]
    return {
        "method": "LASSO squared error on dancer=1/non_dancer=0; video-group CV",
        "solver": "NumPy coordinate descent v1",
        "feature_ids": [PREDICTOR_IDS[i] for i in selected],
        "ranking_coefficients": coef[order].tolist(),
        "coefficient": top_coef.tolist(),
        "intercept": top_intercept,
        "medians": top_medians.tolist(),
        "mean": top_mean.tolist(), "std": top_std.tolist(),
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
