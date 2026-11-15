"""实验 08：阅读 ↔ 想象 跨任务语义迁移（特征级）。

问题：同一被试、同一批语义类别，在“阅读”阶段学到的频带功率模式，能否直接
迁移到“想象”阶段？如果能，说明两个任务共享了部分语义相关的 EEG 结构，
比单任务内部的弱解码更有说服力。

做法：取两个任务共有的类别，用全部阅读试次训练 LDA，直接预测全部想象试次；
反向同理。同时给出同一批类别上的任务内 5 折（按 run 分组）基线作为参照。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402
from plan_auto.common import metrics as M  # noqa: E402

IMAGINE_FEATURES = PROJECT_ROOT / "plan_auto/cache/features"
READING_FEATURES = PROJECT_ROOT / "plan_auto/cache/reading_features"
READING_CACHE = PROJECT_ROOT / "plan_auto/cache/reading"
OUT_ROOT = PROJECT_ROOT / "plan_auto/exp08_cross_task/outputs"


def _lda():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def load_task(subject: str, task: str) -> dict:
    if task == "imagine":
        cache = D.load_cache(subject)
        feat_path = IMAGINE_FEATURES / subject / "features.npz"
        class_ids = json.loads((cache["path"] / "meta.json").read_text(encoding="utf-8"))["class_ids"]
    else:
        import pickle

        cache_dir = READING_CACHE / f"{subject}_sf250"
        cache = {
            "eeg": np.load(cache_dir / "eeg.npy", mmap_mode="r"),
            "labels": np.load(cache_dir / "labels.npy"),
            "runs": np.load(cache_dir / "runs.npy"),
            "n_chars": np.load(cache_dir / "n_chars.npy"),
        }
        feat_path = READING_FEATURES / subject / "features.npz"
        class_ids = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))["class_ids"]
    with np.load(feat_path) as cached:
        bp = cached["bp"].copy()
    return {
        "bp": bp,
        "labels": np.asarray(cache["labels"]),
        "runs": np.asarray(cache["runs"]),
        "n_chars": np.asarray(cache["n_chars"]),
        "class_ids": list(class_ids),
    }


def align(imagine: dict, reading: dict) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray]:
    """取两任务共有类别，并返回过滤后的标签与保留掩码。"""
    common = sorted(set(imagine["class_ids"]) & set(reading["class_ids"]))
    imagine_lookup = {cid: i for i, cid in enumerate(common)}
    reading_lookup = {cid: i for i, cid in enumerate(common)}
    mask_i = np.asarray([imagine["class_ids"][l] in imagine_lookup for l in imagine["labels"]])
    mask_r = np.asarray([reading["class_ids"][l] in reading_lookup for l in reading["labels"]])
    y_i = np.asarray([imagine_lookup[imagine["class_ids"][l]] for l in imagine["labels"][mask_i]])
    y_r = np.asarray([reading_lookup[reading["class_ids"][l]] for l in reading["labels"][mask_r]])
    return y_i, y_r, common, mask_i, mask_r


def cross_task_run_split(
    train_x, y_train, train_runs, test_x, y_test, test_runs, n_splits=5, seed=42
) -> dict:
    """跨任务 + 跨 run：训练任务和测试任务的 run 完全不重叠，排除“同一试次
    的阅读段与想象段共享慢漂移”这种配对泄漏。"""
    folds = D.grouped_folds(test_runs, y_test, n_splits=n_splits, seed=seed)
    pred = np.full(len(y_test), -1)
    for tr, te in folds:
        keep_runs = set(np.unique(test_runs[tr]).tolist())
        fit_mask = np.isin(train_runs, list(keep_runs))
        if fit_mask.sum() < 20:
            continue
        model = _lda()
        model.fit(train_x[fit_mask], y_train[fit_mask])
        pred[te] = model.predict(test_x[te])
    valid = pred >= 0
    return {
        "accuracy": float(np.mean(pred[valid] == y_test[valid])),
        "chance": 1.0 / (int(y_test.max()) + 1),
        "n": int(valid.sum()),
    }


def eval_model(train_x, train_y, test_x, test_y, runs=None, matched_mask=None) -> dict:
    model = _lda()
    model.fit(train_x, train_y)
    prob = model.predict_proba(test_x)
    pred = prob.argmax(1)
    n_classes = int(max(train_y.max(), test_y.max())) + 1
    out = {
        "accuracy": float(np.mean(pred == test_y)),
        "balanced_accuracy": float(M.classification_metrics(test_y, pred, n_classes)["balanced_accuracy"]),
        "chance": 1.0 / n_classes,
        "n": int(len(test_y)),
    }
    if runs is not None:
        _, low, high = M.cluster_bootstrap_ci(test_y, pred, np.asarray(runs))
        out.update({"accuracy_ci_low": low, "accuracy_ci_high": high})
    if matched_mask is not None and matched_mask.any():
        out["matched_accuracy"] = float(np.mean(pred[matched_mask] == test_y[matched_mask]))
        out["matched_n"] = int(matched_mask.sum())
    return out


def within_task_cv(x, y, runs, n_splits=5, seed=42) -> dict:
    folds = D.grouped_folds(runs, y, n_splits=n_splits, seed=seed)
    pred = np.full(len(y), -1)
    for tr, te in folds:
        model = _lda()
        model.fit(x[tr], y[tr])
        pred[te] = model.predict(x[te])
    return {
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(
            M.classification_metrics(y, pred, int(y.max()) + 1)["balanced_accuracy"]
        ),
        "chance": 1.0 / (int(y.max()) + 1),
        "n": int(len(y)),
    }


def run_subject(subject: str) -> dict:
    imagine = load_task(subject, "imagine")
    reading = load_task(subject, "reading")
    y_i, y_r, common, mask_i, mask_r = align(imagine, reading)
    bp_i, bp_r = imagine["bp"][mask_i], reading["bp"][mask_r]
    runs_i, runs_r = imagine["runs"][mask_i], reading["runs"][mask_r]
    rows = []

    def add(name, result, **extra):
        rows.append({"subject": subject, "setting": name, "n_common_classes": len(common), **result, **extra})
        print(f"[{subject}] {name}: acc={result['accuracy']:.3f} chance={result['chance']:.3f}", flush=True)

    add("read->imagine_paired", eval_model(bp_r, y_r, bp_i, y_i, runs=runs_i))
    add("imagine->read_paired", eval_model(bp_i, y_i, bp_r, y_r, runs=runs_r))
    add(
        "read->imagine_runsplit",
        cross_task_run_split(bp_r, y_r, runs_r, bp_i, y_i, runs_i),
    )
    add(
        "imagine->read_runsplit",
        cross_task_run_split(bp_i, y_i, runs_i, bp_r, y_r, runs_r),
    )
    add("imagine_cv", within_task_cv(bp_i, y_i, runs_i))
    add("read_cv", within_task_cv(bp_r, y_r, runs_r))
    return {"rows": rows, "common_classes": common}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for subject in D.SUBJECTS:
        all_rows.extend(run_subject(subject)["rows"])
    table = pd.DataFrame(all_rows)
    table.to_csv(OUT_ROOT / "results.csv", index=False)

    pooled = []
    for setting, part in table.groupby("setting"):
        pooled.append(
            {
                "setting": setting,
                "mean_accuracy": float(part.accuracy.mean()),
                "sd_accuracy": float(part.accuracy.std()),
                "mean_chance": float(part.chance.mean()),
                "n_subjects": len(part),
                "subjects_above_chance": int((part.accuracy > part.chance).sum()),
            }
        )
    pool_table = pd.DataFrame(pooled).sort_values("mean_accuracy", ascending=False)
    pool_table.to_csv(OUT_ROOT / "pooled_results.csv", index=False)
    print(pool_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
