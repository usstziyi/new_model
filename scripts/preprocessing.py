"""Chisco-1.0 EEG 预处理脚本。

在项目根目录用 uv 运行::

    # continuous：连续 raw + 文本 sidecar
    uv run scripts/preprocessing.py --subject sub-01 --ica-mode continuous --n-jobs 4

    # per-condition：read / imagine epoch
    uv run scripts/preprocessing.py --subject sub-01 --ica-mode per-condition --n-jobs 4

    # 多个被试
    uv run scripts/preprocessing.py --subject 01 02 03 --ica-mode per-condition --n-jobs 8

    # 全部被试（默认 continuous）
    uv run scripts/preprocessing.py --ica-mode continuous

    # 查看参数
    uv run scripts/preprocessing.py --help

``--n-jobs`` 同时影响 continuous 与 per-condition 模式下的 AutoReject。
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from autoreject import AutoReject, get_rejection_threshold
from mne_icalabel import label_components
from pyprep.prep_pipeline import PrepPipeline


# ============================================================
# Paths / constants
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONTAGE_FILE = Path(__file__).resolve().parent / "montage.csv"
DATASETS_FOLDER = PROJECT_ROOT / "datasets"
CHISCO1_FOLDER = DATASETS_FOLDER / "ds005170-download"
PREPROCESSED_FOLDER = DATASETS_FOLDER / "preprocessed"
TEXTDATASET = CHISCO1_FOLDER / "textdataset"

# 排除通道
MONTAGE_EXCLUDE_CHANNELS = {"10", "11", "111", "110", "84", "85", "EKG", "EMG", "VEO", "HEO"}
# 辅助通道
STIM_CHANNEL = "Trigger"

# Chisco-1.0 试次起点
TRIAL_EVENT_CODE = 65380
# read / imagine 两段（与原始 Chisco-1.0 脚本一致）
CONDITIONS = (
    ("read", 0.0, 5.0),
    ("imagine", 5.0, 8.3),
)

ICA_MODE_CONTINUOUS = "continuous"
ICA_MODE_PER_CONDITION = "per-condition"


# ============================================================
# Small helpers
# ============================================================
def to_str_list(values):
    """PyPREP 可能返回 numpy.str_，统一转为 Python str。"""
    if values is None:
        return []
    return [str(value) for value in values]


def normalize_subject(subject: str) -> str:
    """统一 subject 格式。

    支持:
        "1"      -> "sub-01"
        "01"     -> "sub-01"
        "sub-01" -> "sub-01"
    """
    subject = subject.strip()
    if subject.startswith("sub-"):
        number = subject.removeprefix("sub-")
    else:
        number = subject
    try:
        return f"sub-{int(number):02d}"
    except ValueError as e:
        raise ValueError(
            f"非法 subject: {subject!r}，应类似 '01'、'1' 或 'sub-01'"
        ) from e


def all_subjects() -> list[str]:
    if not CHISCO1_FOLDER.exists():
        raise FileNotFoundError(f"数据目录不存在: {CHISCO1_FOLDER}")

    subjects = [p.name for p in CHISCO1_FOLDER.glob("sub-*") if p.is_dir()]

    def subject_number(name: str) -> int:
        match = re.fullmatch(r"sub-(\d+)", name)
        if match is None:
            return 10**9
        return int(match.group(1))

    return sorted(subjects, key=subject_number)


EEG_FILENAME_PATTERN = re.compile(
    r"^(?P<subject>sub-\d+)_(?P<session>ses-\d+)_task-(?P<task>[A-Za-z0-9]+)"
    r"_run-(?P<run>\d+)_eeg\.edf$"
)


def parse_eeg_filename(edf_file: Path) -> dict[str, str]:
    """从 EDF 文件名解析 subject / session / task / run。"""
    match = EEG_FILENAME_PATTERN.match(edf_file.name)
    if match is None:
        raise ValueError(f"无法解析 EDF 文件名: {edf_file.name}")
    return match.groupdict()


def iter_subject_recordings(subject: str):
    """遍历某被试全部 session 下的 EDF，按 session、run 编号升序。

    处理入口只到 subject 级别，session 和 task 由文件名内部发现。
    """
    subject_dir = CHISCO1_FOLDER / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"被试目录不存在: {subject_dir}")

    session_dirs = sorted(
        subject_dir.glob("ses-*"),
        key=lambda path: int(path.name.split("-")[1]),
    )
    for session_dir in session_dirs:
        eeg_dir = session_dir / "eeg"
        if not eeg_dir.exists():
            raise FileNotFoundError(f"EEG 目录不存在: {eeg_dir}")

        edf_files = sorted(
            eeg_dir.glob("*_eeg.edf"),
            key=lambda path: int(parse_eeg_filename(path)["run"]),
        )
        for edf_file in edf_files:
            yield parse_eeg_filename(edf_file), edf_file


def recording_prefix(entities: dict[str, str]) -> str:
    """BIDS 风格的文件前缀，不含模态后缀。"""
    return (
        f"{entities['subject']}_{entities['session']}"
        f"_task-{entities['task']}_run-{entities['run']}"
    )


def load_sentences_for_run(run: str) -> list[str]:
    """读取某个 run 对应的句子列表（xlsx 第一列，行数与事件数配套）。"""
    text_file = TEXTDATASET / f"split_data_{int(run)}.xlsx"
    if not text_file.exists():
        raise FileNotFoundError(f"文本文件不存在: {text_file}")
    return pd.read_excel(text_file).iloc[:, 0].astype(str).tolist()


def align_sentences_for_events(
    sentences: list[str],
    n_events: int,
    run: str,
) -> list[str]:
    """把句子数对齐到事件数：文本多则截断，少则报错。"""
    if len(sentences) > n_events:
        print(
            f"[Text] 文本 {len(sentences)} 条，事件 {n_events} 个，"
            f"截断文本到前 {n_events} 条"
        )
        return sentences[:n_events]
    if len(sentences) < n_events:
        raise RuntimeError(
            f"文本数少于事件数: text={len(sentences)}, events={n_events}, run={run}"
        )
    return sentences


def write_text_sidecar(
    text_path: Path,
    trial_onsets,
    sentences: list[str],
):
    """写出 trial_index / onset / Word 的 sidecar TSV。"""
    text_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trial_index": np.arange(len(sentences)),
            "onset": np.asarray(trial_onsets, dtype=float)[: len(sentences)],
            "Word": sentences,
        }
    ).to_csv(text_path, sep="\t", index=False)
    print(f"[Saved] {text_path}")


# ============================================================
# Shared preprocessing
# ============================================================
def prepare_raw(
    edf_file: Path,
    sample_rate: float = 250.0,
):
    """读取 EDF，提取 Trigger 试次，重采样并完成 PyPREP + 1 Hz 高通。

    返回:
        raw:          预处理后的连续 Raw（Trigger 通道已移除）
        trial_onsets: 每个 65380 试次起点的秒级 onset
    """
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    custom_montage = mne.channels.read_custom_montage(MONTAGE_FILE)
    raw.set_montage(custom_montage, on_missing="raise", verbose=False)
    raw.drop_channels(MONTAGE_EXCLUDE_CHANNELS)

    # 事件在重采样前从 Trigger 通道提取；换算成秒后与采样率无关
    events = mne.find_events(raw, stim_channel=STIM_CHANNEL, verbose=False)
    trial_events = events[events[:, 2] == TRIAL_EVENT_CODE]
    trial_onsets = trial_events[:, 0] / raw.info["sfreq"]
    print(f"[Events] trial events ({TRIAL_EVENT_CODE}): {len(trial_onsets)}")

    # 后续不再需要 Trigger；移除后避免混入 epoch / ICA
    raw.drop_channels([STIM_CHANNEL])

    if sample_rate is not None and raw.info["sfreq"] != sample_rate:
        raw.resample(sample_rate, verbose=False)

    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(50, raw.info["sfreq"] / 2, 50),
    }
    print("[PREP] Running PyPREP ...")
    prep = PrepPipeline(raw, prep_params, custom_montage, ransac=False)
    prep.fit()
    raw = prep.raw.copy()

    noisy_original = to_str_list(prep.noisy_channels_original.get("bad_all", []))
    interpolated = to_str_list(prep.interpolated_channels)
    still_bad = to_str_list(prep.still_noisy_channels)
    print(f"[PREP] original bad : {noisy_original}")
    print(f"[PREP] interpolated : {interpolated}")
    print(f"[PREP] still bad    : {still_bad}")
    print(f"[PREP] info['bads']  : {raw.info['bads']}")

    raw.set_annotations(
        mne.Annotations(
            onset=trial_onsets,
            duration=np.zeros(len(trial_onsets), dtype=float),
            description=["trial_start"] * len(trial_onsets),
        )
    )

    # PyPREP 输出仍保留低频漂移，ICA / AutoReject 之前做 1 Hz 高通
    raw.filter(l_freq=1.0, h_freq=None, picks="eeg", verbose=False)
    print("[Filter] 1 Hz high-pass applied to EEG.")
    return raw, trial_onsets


def fit_and_apply_ica(inst, tag: str, apply_inst=None):
    """在 inst 上拟合 ICA，用 ICLabel 剔除伪迹后应用到 apply_inst（默认 inst）。"""
    prefix = f"[ICA:{tag}]"
    ica = mne.preprocessing.ICA(
        n_components=30,
        random_state=97,
        max_iter="auto",
        method="infomax",
        fit_params=dict(extended=True),
    )
    print(f"{prefix} fitting ...")
    ica.fit(inst, picks="eeg", verbose=False)

    ic_labels = label_components(
        inst.copy().pick(ica.ch_names), ica, method="iclabel"
    )
    labels = ic_labels["labels"]
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    print(f"{prefix} component labels : {dict(Counter(labels))}")
    print(f"{prefix} excluding idx    : {exclude_idx}")

    ica.apply(inst if apply_inst is None else apply_inst, exclude=exclude_idx)


def clean_continuous_raw(raw, n_jobs: int = 1):
    """在固定长度 2 s epoch 上用 AutoReject 清洗，ICA 拟合后应用到连续 raw。"""
    fixed_epoch_duration = 2.0
    fixed_events = mne.make_fixed_length_events(
        raw,
        id=1,
        start=0,
        stop=None,
        duration=fixed_epoch_duration,
        overlap=0.0,
    )
    fixed_epochs = mne.Epochs(
        raw,
        events=fixed_events,
        event_id={"fixed": 1},
        tmin=0.0,
        tmax=fixed_epoch_duration - 1.0 / raw.info["sfreq"],
        baseline=None,
        picks="eeg",
        preload=True,
        reject=None,
        flat=None,
        verbose=False,
    )
    n_fixed_before = len(fixed_epochs)
    try:
        auto_reject = AutoReject(n_jobs=n_jobs)
        fixed_epochs, reject_log = auto_reject.fit_transform(
            fixed_epochs, return_log=True
        )
        dropped = int(reject_log.bad_epochs.sum())
    except Exception as exc:
        print(f"[Autoreject] AutoReject 失败（{exc}），回退 get_rejection_threshold")
        reject_threshold = get_rejection_threshold(
            fixed_epochs, ch_types=["eeg"], verbose=False
        )
        fixed_epochs.drop_bad(reject=reject_threshold, verbose=False)
        dropped = n_fixed_before - len(fixed_epochs)

    print(
        f"[Autoreject] epochs kept : {len(fixed_epochs)}/{n_fixed_before} "
        f"(dropped {dropped})"
    )
    if len(fixed_epochs) == 0:
        raise RuntimeError("固定长度 epoch 全部被 AutoReject 丢弃，无法拟合 ICA")

    # ICA 在清洗后的固定长度 epoch 上拟合，再应用到连续 raw
    fit_and_apply_ica(fixed_epochs, tag="continuous", apply_inst=raw)

    # ICA 之后，用已去伪迹的邻近通道插值 still-bad 通道
    if raw.info["bads"]:
        print(f"[ICA] interpolate bads : {raw.info['bads']}")
        raw.interpolate_bads(verbose=False)
    return raw


def epoch_condition(
    raw,
    trial_onsets,
    sentences: list[str],
    phase: str,
    tmin: float,
    tmax: float,
    n_jobs: int,
):
    """按条件切分，AutoReject 后用 ICLabel 清洗并返回 Epochs。"""
    metadata = pd.DataFrame({"Word": sentences})
    sfreq = raw.info["sfreq"]
    onset_samples = np.round(np.asarray(trial_onsets) * sfreq).astype(int)
    events = np.column_stack(
        [
            onset_samples,
            np.zeros(len(onset_samples), dtype=int),
            np.full(len(onset_samples), TRIAL_EVENT_CODE, dtype=int),
        ]
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"trial": TRIAL_EVENT_CODE},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        metadata=metadata,
        verbose=False,
    )
    print(f"[{phase}] epochs: {len(epochs)} ({tmin}-{tmax} s)")
    if len(epochs) == 0:
        raise RuntimeError(f"[{phase}] 没有可切分的 epoch")

    try:
        auto_reject = AutoReject(n_jobs=n_jobs)
        epochs_clean, reject_log = auto_reject.fit_transform(epochs, return_log=True)
        dropped = int(reject_log.bad_epochs.sum())
    except Exception as exc:
        print(f"[{phase}] AutoReject 失败（{exc}），回退 get_rejection_threshold")
        threshold = get_rejection_threshold(epochs, ch_types=["eeg"], verbose=False)
        n_before = len(epochs)
        epochs_clean = epochs.copy()
        epochs_clean.drop_bad(reject=threshold, verbose=False)
        dropped = n_before - len(epochs_clean)

    print(f"[{phase}] kept: {len(epochs_clean)}/{len(epochs)} (dropped {dropped})")
    if len(epochs_clean) == 0:
        raise RuntimeError(f"[{phase}] AutoReject 后没有剩余 epoch")

    fit_and_apply_ica(epochs_clean, tag=phase)

    if epochs_clean.info["bads"]:
        print(f"[{phase}] interpolate bads : {epochs_clean.info['bads']}")
        epochs_clean.interpolate_bads(verbose=False)
    return epochs_clean


# ============================================================
# Output branches
# ============================================================
def convert_edf_to_raw_fif(
    edf_file: Path,
    output_path: Path,
    text_path: Path,
    run: str,
    sample_rate: float = 250.0,
    n_jobs: int = 1,
):
    """continuous 模式：连续 raw 清洗后保存 raw fif，并写句子 sidecar。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw, trial_onsets = prepare_raw(edf_file, sample_rate)

    sentences = align_sentences_for_events(
        load_sentences_for_run(run), len(trial_onsets), run
    )

    clean_continuous_raw(raw, n_jobs=n_jobs)

    write_text_sidecar(text_path, trial_onsets, sentences)
    raw.save(output_path, overwrite=True, verbose=False)
    print(f"[Saved] {output_path}")


def convert_edf_to_epoch_fif(
    edf_file: Path,
    output_dir: Path,
    entities: dict[str, str],
    sample_rate: float = 250.0,
    n_jobs: int = 1,
):
    """per-condition 模式：read / imagine 各自清洗后保存 epoch fif。"""
    raw, trial_onsets = prepare_raw(edf_file, sample_rate)
    if len(trial_onsets) == 0:
        raise RuntimeError(f"没有 {TRIAL_EVENT_CODE} 试次事件: {edf_file}")

    sentences = align_sentences_for_events(
        load_sentences_for_run(entities["run"]), len(trial_onsets), entities["run"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = recording_prefix(entities)
    for phase, tmin, tmax in CONDITIONS:
        epochs_clean = epoch_condition(
            raw=raw,
            trial_onsets=trial_onsets,
            sentences=sentences,
            phase=phase,
            tmin=tmin,
            tmax=tmax,
            n_jobs=n_jobs,
        )
        output_path = output_dir / f"{prefix}_{phase}_epo.fif"
        epochs_clean.save(output_path, overwrite=True, verbose=False)
        print(f"[Saved] {output_path}")


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Chisco-1.0 EEG 预处理（PyPREP + AutoReject + ICA）。"
    )
    parser.add_argument(
        "--subject",
        nargs="+",
        help="被试 ID（例如 01、sub-01，可传多个：--subject sub-01 sub-02）；不传则处理全部被试",
    )
    parser.add_argument(
        "--ica-mode",
        choices=[ICA_MODE_CONTINUOUS, ICA_MODE_PER_CONDITION],
        default=ICA_MODE_CONTINUOUS,
        help=(
            "continuous：连续 raw 上拟合 ICA，输出 raw fif 和 text.tsv；"
            "per-condition：read/imagine 分条件 AutoReject + ICA，输出 epoch fif"
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=250.0,
        help="重采样目标采样率 Hz（默认 250）",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="AutoReject 并行任务数（两种模式都生效，默认 1）",
    )
    args = parser.parse_args()

    subjects = (
        [normalize_subject(subject) for subject in args.subject]
        if args.subject
        else all_subjects()
    )

    for subject in subjects:
        if args.ica_mode == ICA_MODE_CONTINUOUS:
            output_dir = (
                PREPROCESSED_FOLDER / "fif" / f"{args.sample_rate:g}hz" / subject
            )
        else:
            output_dir = (
                PREPROCESSED_FOLDER / "fif-epo" / f"{args.sample_rate:g}hz" / subject
            )

        for entities, edf_file in iter_subject_recordings(subject):
            prefix = recording_prefix(entities)
            print(
                f"[{subject}] {entities['session']} / task-{entities['task']} "
                f"/ run-{entities['run']}: {edf_file.name}"
            )

            if args.ica_mode == ICA_MODE_CONTINUOUS:
                output_paths = [
                    output_dir / f"{prefix}_raw.fif",
                    output_dir / f"{prefix}_text.tsv",
                ]
            else:
                output_paths = [
                    output_dir / f"{prefix}_{phase}_epo.fif"
                    for phase, _, _ in CONDITIONS
                ]

            if all(path.exists() for path in output_paths):
                print(f"[Skip] Existing outputs: {[path.name for path in output_paths]}")
                continue

            if args.ica_mode == ICA_MODE_CONTINUOUS:
                convert_edf_to_raw_fif(
                    edf_file=edf_file,
                    output_path=output_paths[0],
                    text_path=output_paths[1],
                    run=entities["run"],
                    sample_rate=args.sample_rate,
                    n_jobs=args.n_jobs,
                )
            else:
                convert_edf_to_epoch_fif(
                    edf_file=edf_file,
                    output_dir=output_dir,
                    entities=entities,
                    sample_rate=args.sample_rate,
                    n_jobs=args.n_jobs,
                )


if __name__ == "__main__":
    main()
