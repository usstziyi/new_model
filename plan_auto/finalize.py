"""汇总实验：把已训练好的 OOF 概率组合成最终模型，并做显著性检验。

只做概率层面的事后组合，不重新训练，因此每个试次的预测仍然来自“没见过该
试次所属 run”的模型。候选：
* ``feature_ens``：exp06 中 chan / band / hjorth / bp_lda / ensemble_all 的平均
* ``deep``：exp02 的 ShallowFBCSP-OOF
* ``naive_ens``：feature_ens 与 deep 等权平均（注意：该结果受“类别按时间块
  顺序呈现”的混杂影响，不能当作语义解码证据，详见 REPORT.md）
* ``chan``：仅通道组集成

显著性：
* 按 run 的 cluster bootstrap 置信区间（同一 run 内试次相关）
* run 内标签置换检验（保持每个 run 的标签构成，只破坏试次对应）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_auto.common import data as D  # noqa: E402
from plan_auto.common import metrics as M  # noqa: E402

EXP06 = PROJECT_ROOT / "plan_auto/exp06_sweep/outputs"
SHALLOW = PROJECT_ROOT / "plan_auto/exp02_deep/outputs/shallowfbcsp"
OUT = PROJECT_ROOT / "plan_auto/final_model"
FIG = PROJECT_ROOT / "plan_auto/figures"


def _load_imagine(subject: str) -> dict:
    exp06 = np.load(EXP06 / subject / "oof.npz")
    deep = np.load(SHALLOW / subject / "oof.npz")
    labels = exp06["y_true"]
    assert np.array_equal(labels, deep["y_true"]), f"{subject} 标签不一致"
    runs = exp06["runs"]
    matched = exp06["matched_mask"]
    return {"labels": labels, "runs": runs, "matched": matched, "exp06": exp06, "deep": deep}


def _norm(prob: np.ndarray) -> np.ndarray:
    return prob / np.maximum(prob.sum(axis=1, keepdims=True), 1e-12)


def candidate_probs(data: dict) -> dict[str, np.ndarray]:
    exp06, deep = data["exp06"], data["deep"]
    feature = _norm(
        np.mean([_norm(exp06[f"prob_{k}"].astype(np.float64)) for k in ("chan", "band", "hjorth_lda", "bp_lda", "ensemble_all")], axis=0)
    )
    deep_prob = _norm(deep["prob"].astype(np.float64))
    return {
        "chan": _norm(exp06["prob_chan"].astype(np.float64)),
        "feature_ens": feature,
        "deep": deep_prob,
        "naive_ens": _norm(0.5 * feature + 0.5 * deep_prob),
    }


def permutation_p_within_run(y_true, y_pred, runs, n_perm=1000, seed=42) -> float:
    """run 内置换标签的零分布 p 值（向量化实现）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    runs = np.asarray(runs)
    unique, inverse = np.unique(runs, return_inverse=True)
    observed = float(np.mean(y_true == y_pred))

    # 每个 run 的标签与预测分块
    order = np.argsort(inverse, kind="stable")
    splits = np.cumsum(np.bincount(inverse, minlength=len(unique)))
    true_blocks = np.split(y_true[order], splits[:-1])
    pred_blocks = np.split(y_pred[order], splits[:-1])
    n_trials = np.asarray([len(b) for b in true_blocks])

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        correct = 0
        for true_b, pred_b in zip(true_blocks, pred_blocks):
            correct += int(np.sum(rng.permutation(true_b) == pred_b))
        null[i] = correct / len(y_true)
    return float((np.sum(null >= observed) + 1) / (n_perm + 1))


def pool(task: str) -> pd.DataFrame:
    rows = []
    per_subject = []
    pooled_probs: dict[str, list[np.ndarray]] = {}
    y_true, runs, matched = [], [], []

    for subject in D.SUBJECTS:
        data = _load_imagine(subject)
        y_true.append(data["labels"])
        runs.append(np.char.add(f"{subject}:", data["runs"].astype(str)))
        matched.append(data["matched"])
        for name, prob in candidate_probs(data).items():
            pooled_probs.setdefault(name, []).append(prob)

    y_true = np.concatenate(y_true)
    runs = np.concatenate(runs)
    matched = np.concatenate(matched)

    for name, probs in pooled_probs.items():
        prob = np.concatenate(probs)
        pred = prob.argmax(1)
        m = M.classification_metrics(y_true, pred, len(D.CLASS_IDS))
        _, low, high = M.cluster_bootstrap_ci(y_true, pred, runs)
        _, mlow, mhigh = M.cluster_bootstrap_ci(y_true[matched], pred[matched], runs[matched])
        m.update(
            {
                "candidate": name,
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "matched_accuracy": float(np.mean(pred[matched] == y_true[matched])),
                "matched_n": int(matched.sum()),
                "matched_ci_low": mlow,
                "matched_ci_high": mhigh,
                "perm_p_matched": permutation_p_within_run(
                    y_true[matched], pred[matched], runs[matched], n_perm=1000
                ),
            }
        )
        rows.append(m)

        # 每个被试的 matched 准确率
        offset = 0
        for subject in D.SUBJECTS:
            n = int(np.sum(np.char.startswith(runs, f"{subject}:")))
            part_mask = matched[offset:offset + n]
            part_true = y_true[offset:offset + n]
            part_pred = pred[offset:offset + n]
            per_subject.append(
                {
                    "candidate": name,
                    "subject": subject,
                    "accuracy": float(np.mean(part_pred == part_true)),
                    "matched_accuracy": float(np.mean(part_pred[part_mask] == part_true[part_mask])),
                }
            )
            offset += n

    result = pd.DataFrame(rows).sort_values("matched_accuracy", ascending=False)
    result.to_csv(OUT / f"candidates_{task}.csv", index=False)
    pd.DataFrame(per_subject).to_csv(OUT / f"per_subject_{task}.csv", index=False)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pool("imagine")
    print(table[["candidate", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
                 "matched_accuracy", "matched_ci_low", "matched_ci_high", "perm_p_matched"]].to_string(index=False))

    # 保存 naive_ens 的合并 OOF，供画图/复现
    probs, labels, runs, matched = [], [], [], []
    for subject in D.SUBJECTS:
        data = _load_imagine(subject)
        probs.append(candidate_probs(data)["naive_ens"])
        labels.append(data["labels"])
        runs.append(np.char.add(f"{subject}:", data["runs"].astype(str)))
        matched.append(data["matched"])
    np.savez_compressed(
        OUT / "final_oof.npz",
        prob=np.concatenate(probs),
        y_true=np.concatenate(labels),
        runs=np.concatenate(runs),
        matched_mask=np.concatenate(matched),
    )


if __name__ == "__main__":
    main()
