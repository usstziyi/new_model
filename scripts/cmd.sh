# 单被试烟测（置换关掉，确认输出结构）
uv run experiments/diagnostics/baseline_lda/train.py --subject sub-01 --permutations 0 --run-name _smoke
uv run experiments/diagnostics/run_identity/train.py --subject sub-01 --permutations 0 --run-name _smoke
uv run experiments/diagnostics/loro_lodo/train.py --subject sub-01 --permutations 0 --run-name _smoke

# 全量主结果（5 被试，统一 run-name）
# baseline：2000 次置换
uv run experiments/diagnostics/baseline_lda/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --permutations 2000 --run-name main_8_2

# LORO + LODO：2000 次置换（LORO 45 折，成本较高）
uv run experiments/diagnostics/loro_lodo/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --leave-out both --permutations 2000 --run-name main_8_2

# LORO/LODO 分被试续跑（sub-01 已完成；run-name 带被试子目录，避免 metrics.json 互相覆盖）
uv run experiments/diagnostics/loro_lodo/train.py --subject sub-01 --leave-out both --permutations 2000 --run-name main_8_2/sub-01
uv run experiments/diagnostics/loro_lodo/train.py --subject sub-02 --leave-out both --permutations 2000 --run-name main_8_2/sub-02
uv run experiments/diagnostics/loro_lodo/train.py --subject sub-03 --leave-out both --permutations 2000 --run-name main_8_2/sub-03
uv run experiments/diagnostics/loro_lodo/train.py --subject sub-04 --leave-out both --permutations 2000 --run-name main_8_2/sub-04
uv run experiments/diagnostics/loro_lodo/train.py --subject sub-05 --leave-out both --permutations 2000 --run-name main_8_2/sub-05

# run identity：2000 次置换
uv run experiments/diagnostics/run_identity/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --permutations 2000 --run-name main_8_2



# 频段扫描
# baseline 频段扫描
uv run experiments/diagnostics/baseline_lda/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --sweep-bands --permutations 0 --run-name main_8_2

# LORO/LODO 频段扫描
uv run experiments/diagnostics/loro_lodo/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --sweep-bands --permutations 0 --run-name main_8_2

# run identity 频段扫描
uv run experiments/diagnostics/run_identity/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --sweep-bands --permutations 0 --run-name main_8_2

# 通道扫描
# run identity 通道扫描（与频段扫描分开跑）
uv run experiments/diagnostics/run_identity/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --sweep-channels --permutations 0 --run-name main_8_2




# 汇总三个实验
uv run outputs/collect_experiment_summary.py `
  --baseline-run outputs/baseline_lda/main_8_2 `
  --run-identity-run outputs/run_identity/main_8_2 `
  --loro-lodo-run outputs/loro_lodo/main_8_2 `
  --include-sweep


# runwise 比较
uv run experiments/mitigation/runwise_inductive/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --run-name main_8_2
uv run experiments/mitigation/runwise_transductive/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --run-name main_8_2

# 汇总
uv run outputs/collect_runwise_paired.py


# deepnet
# eegnet
# 3) 5 名被试 cv5
uv run experiments/deepnet/eegnet/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --protocol cv5 --run-name cv5_eegnet

# 4) 5 名被试，cv5 + LORO + LODO（论文主结果）
uv run experiments/deepnet/eegnet/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --protocol cv5 loro lodo --run-name main_eegnet

# 5) 250 Hz 版本
uv run experiments/deepnet/eegnet/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --target-sfreq 250 --protocol cv5 loro lodo --run-name main_eegnet_250

# 6) run 身份负对照（需完整 top-k）
uv run experiments/deepnet/eegnet/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --protocol runid --run-name runid_eegnet




# freqcnn
# 3) 5 名被试 cv5
uv run experiments/deepnet/freqcnn/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --protocol cv5 --run-name cv5_freqcnn

# 4) 5 名被试，cv5 + LORO + LODO
uv run experiments/deepnet/freqcnn/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --protocol cv5 loro lodo --run-name main_freqcnn

# 5) 250 Hz 版本
uv run experiments/deepnet/freqcnn/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --target-sfreq 250 --protocol cv5 loro lodo --run-name main_freqcnn_250

# 6) run 身份负对照
uv run experiments/deepnet/freqcnn/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --protocol runid --run-name runid_freqcnn