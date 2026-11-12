"""pad/emotion_features.py -- shared HSEmotion-features extractor + linear probe.

The JAFFE script (`scripts/jaffe_linear_probe.py`) proved that:
  - Extracting HSEmotion's penultimate 1408-dim feature vector
  - Training a logistic regression on top
  - Reporting per-image-stratified k-fold CV (paper convention)
  - Reporting subject-stratified k-fold CV (honest cross-subject)
takes ~10 minutes per dataset CPU-only and lifts JAFFE from 0.54 -> 0.93.

This module factors out the shared logic so the same probe runs on
NHFI, EMOTIC, FER2013 with one entry point: `extract_features(items)`
+ `run_linear_probe(features, labels, groups)`.

Items are dicts: {"image_loader": callable -> BGR ndarray, "label": str,
                  "group": str (subject / source for stratification)}.
"""
from __future__ import annotations
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional

REPO = Path(__file__).resolve().parent.parent
HSEMOTION_ONNX = Path.home() / ".hsemotion" / "enet_b2_7.onnx"
HSEMOTION_8_ONNX = Path.home() / ".hsemotion" / "enet_b2_8.onnx"
FEAT_ONNX = REPO / "pad" / "models" / "enet_b2_7_with_features.onnx"
FEAT_8_ONNX = REPO / "pad" / "models" / "enet_b2_8_with_features.onnx"
FEAT_NAME = "onnx::Gemm_1222"   # post-GlobalMaxPool, 1408-dim (same name in both 7 and 8 class)


def ensure_feature_onnx(class_count: int = 7) -> Path:
    """Modify HSEmotion ONNX (7- or 8-class) once to expose the 1408-dim
    penultimate feature vector as a graph output. Cached. Returns the
    path of the modified model."""
    src = HSEMOTION_ONNX if class_count == 7 else HSEMOTION_8_ONNX
    dst = FEAT_ONNX if class_count == 7 else FEAT_8_ONNX
    if dst.is_file():
        return dst
    import onnx
    print(f"[features] modifying {src} -> {dst}")
    model = onnx.load(str(src))
    out_proto = onnx.helper.make_tensor_value_info(
        FEAT_NAME, onnx.TensorProto.FLOAT, [None, 1408])
    model.graph.output.append(out_proto)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    return dst


def hsemotion_preprocess(img_bgr, target_size: int = 260):
    """HSEmotion preprocessing: resize -> BGR->RGB -> ImageNet-normalize -> NCHW."""
    import cv2
    import numpy as np
    img = cv2.resize(img_bgr, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return img.transpose(2, 0, 1)[None].astype(np.float32)


def extract_features(items: Iterable[dict],
                     batch_size: int = 1,
                     verbose: bool = True,
                     class_count: int = 7,
                     tta: bool = False) -> dict:
    """Run modified HSEmotion ONNX on each item, collect features + labels.

    items: iterable of dicts with keys:
        image_loader: callable() -> BGR ndarray (or None to skip)
        label: str (one of the 7 emotion classes)
        group: str (for stratification; subject id, source, etc.)
        meta: optional dict for misc context

    class_count: 7 (enet_b2_7) or 8 (enet_b2_8); selects which ONNX to use.
    tta: if True, also run the horizontally-flipped image and AVERAGE
         the features + logits with the original. Symmetric augmentation
         that's free at training cost (just 2x slower at inference).

    Returns: {"features": [N, 1408], "labels": [N], "groups": [N],
              "label_set": sorted list of unique labels seen,
              "logits": [N, class_count]}
    """
    import cv2
    import numpy as np
    import onnxruntime as ort

    feat_path = ensure_feature_onnx(class_count=class_count)
    sess = ort.InferenceSession(str(feat_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    in_name = sess.get_inputs()[0].name
    feat_idx = out_names.index(FEAT_NAME)
    logits_idx = out_names.index("output")

    feats = []
    labels = []
    groups = []
    logits = []
    n_skip = 0
    items_list = list(items)
    n_total = len(items_list)
    if verbose:
        print(f"[features] extracting from {n_total} items")

    t0 = time.time()
    for i, it in enumerate(items_list):
        try:
            img = it["image_loader"]()
        except Exception:
            n_skip += 1
            continue
        if img is None or img.size == 0:
            n_skip += 1
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        x = hsemotion_preprocess(img)
        outs = sess.run(out_names, {in_name: x})
        feat = outs[feat_idx][0]
        logit = outs[logits_idx][0]
        if tta:
            # Horizontal flip TTA: avg features + logits with the flipped pass.
            # Symmetric augmentation, free at training cost (2x slower at extract).
            x_flip = hsemotion_preprocess(cv2.flip(img, 1))
            outs_flip = sess.run(out_names, {in_name: x_flip})
            feat = (feat + outs_flip[feat_idx][0]) / 2
            logit = (logit + outs_flip[logits_idx][0]) / 2
        feats.append(feat)
        logits.append(logit)
        labels.append(it["label"])
        groups.append(it.get("group", "default"))
        if verbose and (i + 1) % 500 == 0:
            print(f"[features] {i+1}/{n_total}  ({time.time()-t0:.0f}s)", flush=True)
    if verbose:
        print(f"[features] done in {time.time()-t0:.1f}s, "
              f"{len(feats)} extracted, {n_skip} skipped")
    return {
        "features": np.array(feats),
        "labels": np.array(labels),
        "groups": np.array(groups),
        "logits": np.array(logits),
        "label_set": sorted(set(labels)),
    }


def run_linear_probe(features, labels, groups,
                     n_folds: int = 5,
                     do_subject_strat: bool = True,
                     seed: int = 42) -> dict:
    """Run logistic-regression linear probe with two CV protocols.

    Returns: {"per_image_acc": float, "per_image_std": float,
              "per_image_fold_accs": [n_folds],
              "subject_acc": float, "subject_std": float,
              "subject_fold_accs": [n_folds],
              "label_set": sorted list,
              "per_class_per_image": {label: acc},
              "per_class_subject": {label: acc},
              "logits_for_threshold_analysis": passes through}
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    label_set = sorted(set(labels))
    label_idx = {c: i for i, c in enumerate(label_set)}
    y = np.array([label_idx[l] for l in labels])

    # Per-image stratified k-fold (literature convention)
    img_accs = []
    img_per_class_total = defaultdict(int)
    img_per_class_correct = defaultdict(int)
    img_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, te_idx in img_skf.split(features, y):
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=2000, random_state=seed)
        clf.fit(features[tr_idx], y[tr_idx])
        preds = clf.predict(features[te_idx])
        img_accs.append(float((preds == y[te_idx]).mean()))
        for p, t in zip(preds, y[te_idx]):
            img_per_class_total[label_set[t]] += 1
            if p == t:
                img_per_class_correct[label_set[t]] += 1

    out = {
        "per_image_fold_accs": img_accs,
        "per_image_acc": float(np.mean(img_accs)),
        "per_image_std": float(np.std(img_accs)),
        "per_class_per_image": {c: img_per_class_correct[c] / max(img_per_class_total[c], 1)
                                 for c in label_set},
        "per_class_n": {c: img_per_class_total[c] for c in label_set},
        "label_set": label_set,
    }

    # Subject-stratified k-fold (honest cross-subject metric)
    if do_subject_strat:
        unique_groups = sorted(set(groups))
        rng = np.random.default_rng(seed)
        order = list(unique_groups)
        rng.shuffle(order)
        subj_accs = []
        subj_per_class_total = defaultdict(int)
        subj_per_class_correct = defaultdict(int)
        for k in range(n_folds):
            test_g = set(order[k::n_folds])
            tr_idx = np.array([i for i, g in enumerate(groups) if g not in test_g])
            te_idx = np.array([i for i, g in enumerate(groups) if g in test_g])
            if len(te_idx) == 0 or len(tr_idx) == 0:
                continue
            clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                     max_iter=2000, random_state=seed)
            clf.fit(features[tr_idx], y[tr_idx])
            preds = clf.predict(features[te_idx])
            subj_accs.append(float((preds == y[te_idx]).mean()))
            for p, t in zip(preds, y[te_idx]):
                subj_per_class_total[label_set[t]] += 1
                if p == t:
                    subj_per_class_correct[label_set[t]] += 1
        out["subject_fold_accs"] = subj_accs
        out["subject_acc"] = float(np.mean(subj_accs)) if subj_accs else float("nan")
        out["subject_std"] = float(np.std(subj_accs)) if subj_accs else float("nan")
        out["per_class_subject"] = {
            c: subj_per_class_correct[c] / max(subj_per_class_total[c], 1)
            for c in label_set
        }
    return out


def confidence_threshold_analysis(logits, labels,
                                   thresholds=(0.0, 0.5, 0.7, 0.9)) -> dict:
    """For each confidence threshold, report:
       - coverage: fraction of samples above threshold
       - accuracy on the covered subset (vs zero-shot prediction)

    HSEmotion's 7-class output is logits, not softmax. We softmax here.
    Labels are HSEmotion class names; predicted is argmax of logits."""
    import numpy as np
    HSE_LABELS_7 = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    HSE_LABELS_8 = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
    labels_list = HSE_LABELS_7 if logits.shape[1] == 7 else HSE_LABELS_8
    # softmax
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    pred_idx = probs.argmax(axis=1)
    pred = np.array([labels_list[i] for i in pred_idx])
    top_prob = probs.max(axis=1)
    out = {}
    for t in thresholds:
        mask = top_prob >= t
        if mask.sum() == 0:
            out[t] = {"coverage": 0.0, "accuracy": float("nan"), "n": 0}
            continue
        acc = float((pred[mask] == labels[mask]).mean())
        out[t] = {"coverage": float(mask.mean()),
                  "accuracy": acc,
                  "n": int(mask.sum())}
    return out
