"""pad/emotion_eval.py -- shared 5-fold CV evaluation utilities for the
emotion analysis scripts.

Replaces the four separate copies of cv_predictions / cv_classifier /
cv_predictions_with_proba previously living in:
  - scripts/emotion_tier_a_sweep.py        (cv_classifier, returning mean/std/accs)
  - scripts/emotion_confusion_matrices.py  (cv_predictions, preds only)
  - scripts/emotion_balanced_metrics.py    (cv_predictions, preds only)
  - scripts/emotion_deployment_curves.py   (cv_predictions_with_proba)

All three flavours share the same fold split + classifier dispatch +
StandardScaler discipline, so consolidate here.

Also adds `cv_predictions_cached`, which writes/reads predictions to
`reports/_tier_a_cache/preds_<cache_name>_<kind>.npz`. Lets confmat /
balmetric / deploy share the same 5-fold CV result instead of each
re-running it (~5 min savings on the FER2013 SVM-RBF path).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from pad.emotion_config import (
    CV_N_FOLDS, CV_SEED, MLP_HIDDEN_UNITS, SVM_C, LR_C, TIER_A_CACHE_DIR,
)


def _build_classifier(kind: str, seed: int = CV_SEED, *,
                       want_proba: bool = False):
    """Return an unfitted sklearn estimator for one of {lr, mlp, svm}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    if kind == "lr":
        return LogisticRegression(penalty="l2", C=LR_C, solver="lbfgs",
                                   max_iter=2000, random_state=seed)
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(MLP_HIDDEN_UNITS,),
                              activation="relu", solver="adam", alpha=1e-4,
                              max_iter=200, random_state=seed,
                              early_stopping=False)
    if kind == "svm":
        return SVC(kernel="rbf", C=SVM_C, gamma="scale", random_state=seed,
                    probability=want_proba)
    raise ValueError(f"unknown classifier kind: {kind!r}")


def _fold_iter(features, y, n_folds: int = CV_N_FOLDS, seed: int = CV_SEED):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return skf.split(features, y)


def cv_classifier(features, y, *, kind: str = "lr",
                   n_folds: int = CV_N_FOLDS, seed: int = CV_SEED):
    """Run 5-fold per-image stratified CV; return (mean, std, [per-fold accs]).

    Used by the Tier-A sweep where we report headline accuracy + std.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    accs = []
    for tr, te in _fold_iter(features, y, n_folds, seed):
        Xtr, Xte = features[tr], features[te]
        if kind in ("mlp", "svm"):
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = _build_classifier(kind, seed)
        clf.fit(Xtr, y[tr])
        accs.append(float((clf.predict(Xte) == y[te]).mean()))
    return float(np.mean(accs)), float(np.std(accs)), accs


def cv_predictions(features, y, *, kind: str = "lr",
                    n_folds: int = CV_N_FOLDS, seed: int = CV_SEED,
                    return_proba: bool = False):
    """Run 5-fold per-image stratified CV; return aggregated test-fold preds.

    If return_proba=True, also return aggregated max_probs per sample.
    SVM gets `probability=True` (Platt-calibrated) when proba is requested.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    n = len(y)
    all_pred = np.full(n, -1, dtype=int)
    all_prob = np.zeros(n, dtype=np.float32) if return_proba else None
    for tr, te in _fold_iter(features, y, n_folds, seed):
        Xtr, Xte = features[tr], features[te]
        if kind in ("mlp", "svm"):
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = _build_classifier(kind, seed, want_proba=return_proba)
        clf.fit(Xtr, y[tr])
        if return_proba:
            probs = clf.predict_proba(Xte)
            all_pred[te] = probs.argmax(axis=1)
            all_prob[te] = probs.max(axis=1)
        else:
            all_pred[te] = clf.predict(Xte)
    if return_proba:
        return all_pred, all_prob
    return all_pred


def cv_predictions_cached(features, y, *, cache_key: str, kind: str = "lr",
                           n_folds: int = CV_N_FOLDS, seed: int = CV_SEED,
                           return_proba: bool = False, cache_dir: Optional[Path] = None):
    """Same as `cv_predictions`, but caches results to NPZ keyed by
    (cache_key, kind, return_proba). Skips re-running CV on cache hit.

    Three scripts (confmat / balmetric / deploy) each used to re-run the
    same 5-fold CV on the same cached features. With this helper, the
    first one warms the cache and the next two read from disk. ~5 min
    saved on FER2013 SVM.
    """
    import numpy as np
    if cache_dir is None:
        cache_dir = TIER_A_CACHE_DIR
    cache_dir = Path(cache_dir)
    suffix = "_proba" if return_proba else ""
    cache_path = cache_dir / f"preds_{cache_key}_{kind}{suffix}.npz"
    if cache_path.is_file():
        d = np.load(str(cache_path))
        if return_proba:
            return d["pred"], d["prob"]
        return d["pred"]
    out = cv_predictions(features, y, kind=kind, n_folds=n_folds, seed=seed,
                          return_proba=return_proba)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if return_proba:
        np.savez_compressed(str(cache_path), pred=out[0], prob=out[1])
    else:
        np.savez_compressed(str(cache_path), pred=out)
    return out


def load_cached_features(cache_path: Path):
    """Load an emotion feature cache and return (features, labels, label_set, y).

    `y` is the integer-encoded version of `labels` using the sorted unique
    label set. Used by every downstream eval script.
    """
    import numpy as np
    d = np.load(str(cache_path), allow_pickle=True)
    features = d["features"]
    labels = d["labels"]
    label_set = sorted(set(labels.tolist()))
    label_idx = {c: i for i, c in enumerate(label_set)}
    y = np.array([label_idx[l] for l in labels])
    return features, labels, label_set, y
