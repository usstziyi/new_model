# 单被试烟测（置换关掉，确认输出结构）
uv run plan_baseline_lda/train.py --subject sub-01 --permutations 0 --run-name _smoke
uv run plan_run_identity/train.py --subject sub-01 --permutations 0 --run-name _smoke
uv run plan_loro_lodo/train.py --subject sub-01 --permutations 0 --run-name _smoke

# 全量主结果（5 被试，统一 run-name）
# baseline：2000 次置换
uv run plan_baseline_lda/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --permutations 2000 --run-name main_8_2

# LORO + LODO：200 次置换（LORO 45 折，成本较高）
uv run plan_loro_lodo/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --leave-out both --permutations 2000 --run-name main_8_2

# run identity：2000 次置换
uv run plan_run_identity/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --permutations 2000 --run-name main_8_2



# 频段 / 通道扫描
# baseline 频段扫描
uv run plan_baseline_lda/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --sweep-bands --permutations 0 --run-name main_8_2

# LORO/LODO 频段扫描
uv run plan_loro_lodo/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --sweep-bands --permutations 0 --run-name main_8_2

# run identity 频段扫描
uv run plan_run_identity/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --sweep-bands --permutations 0 --run-name main_8_2

# run identity 通道扫描（与频段扫描分开跑）
uv run plan_run_identity/train.py --subject sub-01 sub-02 sub-03 sub-04 sub-05 --sweep-channels --permutations 0 --run-name main_8_2




# 汇总三个实验
uv run outputs/collect_experiment_summary.py `
  --baseline-run outputs/baseline_lda/main_8_2 `
  --run-identity-run outputs/run_identity/main_8_2 `
  --loro-lodo-run outputs/loro_lodo/main_8_2 `
  --include-sweep