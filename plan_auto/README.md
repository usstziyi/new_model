# plan_auto —— Chisco EEG 语义解码实验（夜间批量实验）

目标：在 `datasets/top10_imagine`（及后来的 `top10_reading`）上训练模型，检验
EEG 中是否存在可跨 run 复现的语义类别信息。每个实验独立目录、独立输出，公共
代码只在 `common/`。

**先看结论**：[`REPORT.md`](REPORT.md)。简版：naive 分组 CV 能到 12–13%，
但控制“run 内试次位置/呈现顺序”后所有模型回落到机会水平（run-balanced recall
8.5–9.7%）；训练级置换显示打乱标签后同一流程仍有 11.8%（p=0.90），半合成注入
给出频带功率模型的检测阈约为 SNR 0.5。因此不能声称显著语义解码。

## 目录

| 目录 | 内容 | 输出 |
|---|---|---|
| `common/` | 缓存（0.5/1.5 Hz 高通）、按 run 分组划分、指标、bootstrap、Tee | — |
| `exp00_data_audit/` | 类别/run/句长分布、频带 eta²、句长捷径基线 | `outputs/` |
| `exp01_features/` | 频带功率/Hjorth/ERP/协方差切空间 + 线性模型，嵌套选型 | `outputs/sub-XX/` |
| `exp02_deep/` | ShallowFBCSP / EEGNet，分组 CV、增强、TTA、快照集成 | `outputs/<model>/<subject>/` |
| `exp03_runcenter/` | run 内中心化/标准化对照（未选出，保留） | `outputs/` |
| `exp05_ensemble/` | 早期多特征概率集成 | `outputs/` |
| `exp06_sweep/` | 频带/时间窗/通道组/集成系统扫描（想象） | `outputs/pooled_config_results.csv` |
| `exp07_reading/` | 阅读任务同一套扫描 | `outputs/pooled_config_results.csv` |
| `exp08_cross_task/` | 阅读↔想象迁移：配对 vs 按 run 分离 | `outputs/results.csv` |
| `exp09_confound/` | **决定性实验**：1.5 Hz 高通、位置去趋势、run-balanced recall | `outputs/all_results.csv` |
| `exp10_deep_control/` | 深度模型 + 位置去趋势（非线性对照） | `outputs/results.csv` |
| `exp11_injection/` | 半合成信号注入，估计流程检测阈 | `outputs/pooled_curve.csv` |
| `exp12_permutation/` | 训练级置换检验（run 内打乱标签重训 300 次） | `outputs/results.csv` |
| `finalize.py` | OOF 概率集成 + run 内置换检验 | `final_model/` |
| `make_figures.py` | 报告图 | `figures/` |

## 评估协议（所有实验统一）

1. **按 run 分组 5 折**，同一 run 不进训练/测试两侧。
2. 训练折内再留 25% run 作验证集选配置/早停。
3. 报告：accuracy、macro-F1、句长配平子集、**run 内类别平衡召回**、按 run 的
   cluster bootstrap 95% CI、run 内标签置换 p。
4. 句长配平：每个长度值取各类最小样本数等量抽样，消除“类别 ↔ 句长”捷径。

## 快速复现

```powershell
uv run python plan_auto/exp00_data_audit/audit.py          # 建 250 Hz 缓存 + 审计
uv run python plan_auto/exp06_sweep/run.py --jobs 3        # 想象任务扫描
uv run python plan_auto/exp02_deep/run.py --models shallowfbcsp --epochs 60 --patience 12 --lr 5e-4
uv run python plan_auto/exp08_cross_task/run.py
uv run python plan_auto/exp09_confound/run.py --jobs 2     # 建 1.5 Hz 缓存 + 混杂控制
uv run python plan_auto/exp11_injection/run.py --jobs 2    # 半合成注入
uv run python plan_auto/exp12_permutation/run.py --n-perm 300 --jobs 3
uv run python plan_auto/finalize.py
uv run python plan_auto/make_figures.py
uv run python plan_auto/make_paper_figures.py
```

阅读任务需要先生成 CSV 并导出 top10：

```powershell
uv run python scripts/make_text_csv.py
uv run python scripts/export_top_classes.py --task reading --top-k 10
uv run python plan_auto/exp07_reading/run.py --jobs 2
```

## 缓存说明

- `plan_auto/cache/<subject>_sf250/`：0.5–45 Hz，250 Hz，float16，单位 µV，未 z-score
- `plan_auto/cache/<subject>_sf250_hp1.5/`：1.5–45 Hz，用于混杂控制
- `plan_auto/cache/reading/<subject>_sf250/`：阅读任务
- `plan_auto/cache/features*/`：频带功率、Hjorth、ERP 特征缓存
- sub-05 缺少 `class_14.pkl` 时，缓存构建会回到原始 run pkl 按 CSV 的 `run+pos` 现场抽取

## 下一步建议

见 `REPORT.md` 第 5 节：优先改评估与实验设计（随机化/位置平衡、重复呈现平均、
位置作协变量的混合模型），而不是继续堆模型。
