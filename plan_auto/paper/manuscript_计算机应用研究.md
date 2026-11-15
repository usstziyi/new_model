# 面向想象语音语义解码的呈现顺序混杂诊断与校正方法

**作者**：（待填）　　**单位**：（待填）　　**邮箱**：（待填）

---

## 摘要

**目的**：现有 EEG 想象语音语义解码研究多在随机试次划分下报告高于机会水平的准确率，但该类范式常按语义类别分块呈现，试次在 run 内的位置与类别高度相关，可能高估真实语义信息。**方法**：构建仅使用 run 内试次位置的基线，提出信号级 1.5 Hz 高通与位置多项式去趋势、特征级位置残差化两种校正，并以 run 内类别平衡召回、句长配平、训练级置换检验复核；在 5 名被试、10 类 Chisco 数据上评估频带功率、Hjorth、ERP、ShallowFBCSP 及概率集成，同时通过半合成信号注入校准流程灵敏度。**结果**：仅位置特征即达 35.4% 准确率（10 类）；常规按 run 分组交叉验证为 13.1%，位置去趋势后降至 9.65%，run 内类别平衡召回 8.89%；训练级置换显示打乱标签后同一流程仍得到 11.8%（高于真实标签的 11.3%，p=0.90），去趋势后 p=0.91/0.99；半合成注入表明频带功率流程在信噪比约 0.5 时可稳定检出已知类别信号。**结论**：该数据集上现有模型未提供独立于呈现顺序的语义证据；所提“混杂诊断—校正—复核”框架可用于 EEG 语义解码研究的可信评估，并提示后续实验应随机化或平衡呈现顺序。

**关键词**：脑电；想象语音；语义解码；呈现顺序混杂；评估协议；位置去趋势

**中图分类号**：TP391.4　　**文献标志码**：A

**基金项目**：国家重点研发计划项目（2022YFC3601400）；国家自然科学基金项目（61903255）

---

## Abstract

**Diagnosing and Correcting Presentation-Order Confounding in EEG-Based Semantic Decoding of Imagined Speech**

**Authors**：（to be filled）　　**Affiliation**：（to be filled）

**Abstract**: **Purpose** Existing studies on EEG-based semantic decoding of imagined speech often report above-chance accuracy under random trial-wise splits. However, imagined-speech paradigms usually present semantic categories in blocks within each run, so the trial position is strongly correlated with the category and may inflate the measured performance. **Methods** We quantify this confound on the Chisco dataset (5 subjects, 10 categories). A position-only baseline is constructed using the within-run normalized trial position. Two corrections are proposed: signal-level 1.5-Hz high-pass filtering with per-run polynomial position detrending, and feature-level position residualization. Evaluation is further controlled by run-balanced recall, sentence-length-matched subsets, run-level cluster bootstrap, and training-level permutation tests. Band-power, Hjorth, ERP, ShallowFBCSP and probability ensembles are benchmarked, and the sensitivity of the pipeline is calibrated by semi-synthetic injection of class-specific alpha bursts at controlled signal-to-noise ratios (SNRs). **Results** Position alone achieves 35.4% accuracy (39.5% run-balanced recall) for 10 classes. Conventional run-grouped cross-validation reaches 13.1%, which drops to 9.65% after position detrending (run-balanced recall 8.89%). Training-level permutation shows that shuffling labels within runs yields 11.8% accuracy, even higher than the 11.3% obtained with true labels (p = 0.90); after detrending, p = 0.91/0.99. Semi-synthetic injection indicates that the band-power pipeline reliably detects injected effects at SNR ≈ 0.5, whereas the corrected real-data performance corresponds to SNR ≈ 0. **Conclusion** On this dataset, current models provide no reliable evidence of semantic information independent of presentation order. The proposed diagnose–correct–verify framework offers a practical protocol for trustworthy evaluation of EEG semantic decoding, and highlights the need for randomized or position-balanced stimulus presentation in future data collection.

**Keywords**: electroencephalography (EEG); imagined speech; semantic decoding; presentation-order confounding; evaluation protocol; position detrending

**Foundation item**: National Key Research and Development Program of China (2022YFC3601400); National Natural Science Foundation of China (61903255)

---

## 1 引言

基于脑电（electroencephalography, EEG）的想象语音解码被视为面向自然交互的脑机接口（brain–computer interface, BCI）的重要方向。Chisco 等大规模中文想象语音数据集公开后，围绕语义类别解码的工作明显增多，部分研究报告了 10 类乃至 39 类任务上高于机会水平的准确率[1-3]。这些结果通常被解释为 EEG 中存在可读的语义表征。

然而，评价协议本身可能带来系统性高估。想象语音范式中，同一 run 内的刺激往往按语义类别分块、按固定或半固定顺序呈现：被试先阅读一句提示语，再在约 3.3 s 的空屏期想象该句。这样一来，试次在 run 内的位置与语义类别高度相关，任何随 run 缓慢变化的分量——电极漂移、阻抗变化、疲劳、低频功率、肌电基线——都可能伪装成“语义信息”。随机试次划分会让同一 run 的相邻试次同时进入训练与测试，进一步放大这一效应。数据泄漏与评估偏差在机器学习与医学影像领域已有系统讨论[4-5]，但在想象语音 EEG 文献中尚未得到足够重视。

本文不追求更高的分类准确率，而是回答一个更基础的问题：**在控制呈现顺序后，这些数据里还能读出多少语义信息？** 主要贡献如下：

1. 量化了想象语音数据中的呈现顺序混杂：仅用 run 内归一化位置做 10 类分类，准确率即达 35.4%，run 内类别平衡召回 39.5%；
2. 提出一套“诊断—校正—复核”流程：位置基线、信号级高通与位置去趋势、特征级位置残差化，以及 run 内类别平衡召回、句长配平、训练级置换等复核指标；
3. 在该流程下系统评估 5 类特征模型与 ShallowFBCSP，结合半合成信号注入给出检测阈值，结论是该数据上的语义效应低于可检测水平；同时给出对后续采集与评估的具体建议。

---

## 2 相关工作

### 2.1 想象语音语义解码

本文使用 Chisco 数据集中 5 名被试的想象与阅读任务数据；每名被试约 2 000–2 500 条想象语音试次，原始类别共 39 个，并提供句子文本、类别编号、run 与试次位置等元信息[1]。基于该数据集，已有工作使用卷积网络、EEGNet 类模型或 EEG–文本对比学习进行类别解码或语义对齐[2-3]。这些工作普遍以随机划分或按试次级交叉验证报告 Top-k 准确率，较少讨论 run 内呈现顺序的影响。

### 2.2 EEG 深度学习模型

EEGNet 以紧凑的深度可分离卷积实现跨范式 EEG 分类[6]；ShallowFBCSPNet 将经典 FBCSP 的时频滤波思想与浅层卷积结合，在多个 EEG 解码任务上表现稳健[7]。本文把它们作为深度模型代表，与频带功率、Hjorth、ERP 等手工特征模型对照。

### 2.3 评估偏差与数据泄漏

Kaufman 等系统定义了数据泄漏的形式、检测与规避方法[4]；Varoquaux 与 Cheplygina 指出医学影像机器学习中普遍存在划分不当、预处理泄漏与选择偏差[5]。在 EEG 中，同一 run 内试次共享电极状态与时间漂移，按试次随机划分会破坏独立性假设。本文把这一原则具体化为“位置基线 + 位置去趋势 + run 内置换”的可操作流程。

---

## 3 方法

### 3.1 问题定义

设第 $i$ 个试次的 EEG 为 $x_i\in\mathbb{R}^{C\times T}$，类别标签为 $y_i\in\{0,\dots,9\}$，所属 run 为 $r_i$，在 run 内的归一化位置为

$$
p_i=\frac{\text{pos}(i)}{n_{r_i}-1}\in[0,1],
$$

其中 $\text{pos}(i)$ 为该试次在 run 内的序号，$n_{r_i}$ 为该 run 的试次数。目标是估计条件分布 $P(y\mid x)$，并检验在控制 $p$ 之后 $x$ 是否仍携带类别信息。

### 3.2 位置基线

仅使用位置的多项式特征

$$
\phi(p_i)=\left[p_i,p_i^2,p_i^3,p_i^4\right]
$$

训练线性判别分析（LDA），按 run 分组 5 折交叉验证。该基线不接触任何 EEG，用来衡量“呈现顺序本身能解释多少类别信息”。

### 3.3 信号级校正

对原始信号做 1.5–45 Hz 零相位 Butterworth 带通（4 阶，`sosfiltfilt`），降采样至 250 Hz，并按试次逐通道 z-score。随后对每个 run、每个通道，用 5 阶多项式拟合“电压 ~ 归一化位置”，减去拟合值：

$$
\tilde{x}_{i,c}(t)=x_{i,c}(t)-\sum_{d=0}^{5}\beta_{r_i,c,t,d}\,p_i^d .
$$

该操作移除同一 run 内随位置平滑变化的共模成分，保留同一位置附近试次之间的差异。1.5 Hz 高通与多项式去趋势互为补充：前者抑制慢漂移，后者针对分块呈现造成的台阶。

### 3.4 特征级校正

从校正后的信号提取三类特征：

- **频带功率**：delta（1–4 Hz）、theta（4–8 Hz）、alpha（8–13 Hz）、beta（13–30 Hz）、lowgamma（30–45 Hz），每通道每频带取对数平均功率；
- **Hjorth 参数**：activity、mobility、complexity；
- **ERP 波形**：降采样至 20 Hz 的时域幅值。

对特征矩阵同样按 run 内位置做 3 阶多项式残差化：

$$
\tilde{f}_{i,k}=f_{i,k}-\sum_{d=0}^{3}\gamma_{r_i,k,d}\,p_i^d ,
$$

其中 $k$ 为特征维。该步骤在特征层再次剥离可由位置解释的成分。

### 3.5 可信评估指标

**run 内类别平衡召回（run-balanced recall）**：在每个 run 内计算各类别召回率并取平均，再按 run 试次数加权：

$$
R=\sum_{r}\frac{n_r}{N}\cdot\frac{1}{|\mathcal{C}_r|}\sum_{c\in\mathcal{C}_r}\text{Recall}_{r,c}.
$$

该指标消除 run 内类别比例不均带来的虚高。

**句长配平子集**：对每个句长值，取 10 个类别中该长度样本的最小数量做等量抽样，使各类别句长分布一致，排除“类别 ↔ 句长”捷径。

**按 run 的 cluster bootstrap**：以 run 为独立单位有放回重采样 2 000 次，给出准确率的 95% 置信区间。

**训练级置换检验**：在每个 run 内随机打乱标签（保持该 run 的类别构成与试次数），用相同划分重新训练模型，重复 $B=300$ 次，经验 $p$ 值为

$$
p=\frac{1+\#\{b:\;M_b\ge M_{\text{obs}}\}}{B+1},
$$

其中 $M$ 为准确率或 run-balanced recall。

### 3.6 半合成信号注入

为区分“真实数据没有效应”与“流程检测不到效应”，按真实标签向 EEG 注入类别特异的 alpha 爆发：每个类别固定 30 个通道、8–12 Hz 频率、随机初相与 Hann 包络，幅度按

$$
\text{SNR}=\frac{\text{RMS}_{\text{inj}}}{\text{RMS}_{\text{sig}}}
$$

控制。对每个 SNR 重复完整评估流程，得到“注入效应—可检出性”曲线；将真实数据的结果映射到该曲线，得到其相对效应量的上界。

### 3.7 模型与训练配置

- **特征模型**：标准化 + LDA（`lsqr`，自动收缩）；高维 ERP 先做 96 维 PCA；
- **频带集成**：逐频带 LDA 的概率平均；
- **通道组集成**：额、中央、颞、顶、枕 5 组通道分别建模后平均概率；
- **深度模型**：ShallowFBCSPNet，125 Hz 输入，逐试次逐通道 z-score，训练时做时间平移（±0.2 s）、幅度缩放、高斯噪声、通道丢弃与时间遮挡，测试时 5 个平移做 TTA，按验证集选 top-3 epoch 做快照集成；
- **划分**：5 折按 run 分组，训练折内再留 25% run 作验证集用于选型/早停。

---

## 4 实验与结果

### 4.1 数据与预处理

采用 Chisco 数据集的想象任务（task-imagine）与阅读任务（task-read），数据统计如表 1 所示。取全体被试汇总后样本最多的 10 个语义类别（编号 2, 10, 13, 14, 19, 21, 24, 26, 31, 32），使 5 名被试使用同一套类别。sub-05 缺少的类别 14 从原始 run pkl 按 CSV 的 `run + pos` 现场抽取补齐。

**表 1　数据统计**

| 被试 | 想象试次 | 阅读试次 | run 数 | 类别数 |
|---|---:|---:|---:|---:|
| sub-01 | 2 459 | 2 503 | 45 | 10 |
| sub-02 | 2 524 | 2 515 | 45 | 10 |
| sub-03 | 2 076 | 2 115 | 45 | 10 |
| sub-04 | 2 454 | 2 507 | 45 | 10 |
| sub-05 | 2 122 | 2 042 | 44 | 10 |

注：想象段时长 3.3 s，阅读段 5 s；122 导 EEG；原始采样率 500 Hz，统一降至 250 Hz；类别数为本文使用的 10 类。

### 4.2 呈现顺序混杂的量化

在 sub-01 上，各类别的 run 内平均位置从 0.045 单调升至 0.963，类别与位置的 Spearman 相关系数为 0.91。仅用 4 阶位置多项式做 10 类分类：

- 合并 5 名被试：准确率 **0.354**，run-balanced recall **0.395**；
- 阅读任务同样：位置基线 **0.344–0.377**。

这说明“时间位置”几乎是一个强类别代理。图 1 给出了位置基线、原始 EEG 特征与位置去趋势后 EEG 的 run-balanced recall 对比。

![图 1　仅位置、原始 EEG 与位置去趋势特征的 run-balanced recall 对比](../figures/confound.png)

此外，仅用句长分布做分类的基线为 **12.6%–15.0%**；同一试次的阅读段与想象段配对评估可达 **18.4%–19.9%**，而按 run 分离后降至 **11.9%–12.1%**，说明配对试次共享漂移（图 2）。

![图 2　阅读与想象跨任务迁移在配对评估和按 run 分离评估下的准确率](../figures/cross_task.png)

### 4.3 协议阶梯

以想象任务、5 被试合并为例，逐级收紧评估协议，结果如表 2 所示：

**表 2　评估协议阶梯（10 类，机会水平 10%）**

| 评估协议 | 准确率 | run-balanced recall |
|---|---:|---:|
| 随机试次划分（本组前期复现 ShallowFBCSP） | 0.191 | — |
| 按 run 分组 5 折（通道组集成） | 0.131 | 0.098 |
| + 句长配平子集 | 0.124 | — |
| + 位置去趋势 | 0.097 | 0.089 |
| 仅位置基线（无 EEG） | 0.354 | 0.395 |

注：随机划分结果为同一数据、同一模型在 8∶1∶1 随机试次划分下的复现值，仅作对照；通道组集成为额、中央、颞、顶、枕 5 组通道 LDA 的概率平均；run-balanced recall 为 run 内类别平衡召回；短横线表示该协议下未单独计算。

从 19.1% 到 9.7% 的落差说明：随机划分、run 内类别比例、句长分布与呈现顺序共同贡献了此前的“高性能”。

### 4.4 常规分组 CV 下的模型对比

常规按 run 分组 5 折的模型对比结果如表 3 所示。

**表 3　想象任务常规按 run 分组结果（5 被试合并）**

| 模型/特征 | 准确率 | 95% CI | 句长配平准确率 |
|---|---:|---|---:|
| 通道组集成 | 0.131 | [0.120, 0.143] | 0.124 |
| 频带集成 | 0.122 | [0.109, 0.135] | 0.115 |
| Hjorth + LDA | 0.119 | [0.109, 0.130] | 0.119 |
| 全频带功率 + LDA | 0.118 | [0.108, 0.130] | 0.114 |
| ShallowFBCSP | 0.128 | [0.116, 0.141] | 0.121 |
| ERP + PCA-LDA | 0.108 | [0.101, 0.114] | 0.100 |

注：95% CI 为按 run 的 cluster bootstrap 置信区间；句长配平子集使各类别句长分布一致；所有模型均按 run 分组 5 折评估。

阅读任务的结论类似：Hjorth 的句长配平准确率为 0.129 [0.119, 0.140]，频带集成为 0.124，ERP 为 0.098。阅读并不比想象更容易（图 3、图 4）。

![图 3　想象任务中各特征/模型在句长配平子集上的准确率与按 run 的 95% 置信区间](../figures/imagine_configs.png)

![图 4　阅读任务中各特征/模型在句长配平子集上的准确率与按 run 的 95% 置信区间](../figures/reading_configs.png)

### 4.5 位置校正后的结果

位置校正前后的结果如表 4 所示。

**表 4　1.5 Hz 高通与位置去趋势后的结果（5 被试合并）**

| 设置 | 准确率 | run-balanced recall | 句长配平准确率 |
|---|---:|---:|---:|
| 仅位置（无 EEG） | 0.354 | 0.395 | 0.357 |
| 原始频带集成 | 0.121 | 0.096 | 0.115 |
| 原始全频带 LDA | 0.113 | 0.097 | 0.109 |
| 去趋势频带集成 | 0.097 | 0.085 | 0.091 |
| 去趋势全频带 LDA | 0.097 | 0.089 | 0.094 |
| 去趋势 Hjorth | 0.097 | 0.091 | 0.096 |
| 去趋势 ERP | 0.098 | 0.095 | 0.088 |

注：去趋势指 1.5 Hz 高通后按 run 内位置做多项式残差化（波形 5 阶、特征 3 阶）；“仅位置”基线不使用任何 EEG；表中为 5 名被试合并结果。

深度模型在位置去趋势后的时域波形上训练，run-balanced recall 为 **0.1039**（95% CI [0.0962, 0.1120]），run 内置换 p=0.386。校正后所有模型均与机会水平不可区分，部分指标略低于 10%。

### 4.6 训练级置换检验

训练级置换结果如表 5 所示。

**表 5　训练级置换结果（300 次/被试，run 内打乱标签）**

| 特征 | 指标 | 观测均值 | 零分布均值 | p |
|---|---|---:|---:|---:|
| 原始频带功率 | 准确率 | 0.113 | **0.118** | 0.904 |
| 原始频带功率 | run-balanced recall | 0.097 | 0.103 | 0.937 |
| 位置去趋势 | 准确率 | 0.096 | 0.101 | 0.907 |
| 位置去趋势 | run-balanced recall | 0.089 | 0.100 | 0.993 |

注：每次置换在每个 run 内打乱标签（保持类别构成与试次数），用相同划分重新训练 LDA；p 值为经验单侧 p；零分布均值为 300 次置换的平均。

最值得注意的是：**标签完全打乱后，同一流程仍得到 11.8% 的准确率，高于真实标签的 11.3%**。这表明常规分组 CV 下的 12%–13% 可由 run 内类别构成与位置结构解释，不需要语义信息（图 5）。

![图 5　训练级置换零分布与观测值（run 内打乱标签后重训 300 次）](../figures/permutation_null.png)

### 4.7 半合成注入的灵敏度

不同注入强度下的结果如表 6 所示。

**表 6　注入 SNR 与频带功率模型的准确率（5 被试均值）**

| 注入 SNR | 原始特征 | 位置去趋势 |
|---|---:|---:|
| 0（真实数据） | 0.113 | 0.096 |
| 0.20 | 0.116 | 0.098 |
| 0.30 | 0.128 | 0.106 |
| 0.50 | 0.224 | 0.162 |
| 0.80 | 0.453 | 0.293 |

注：注入为类别特异的 alpha 爆发（每类 30 个通道、8–12 Hz、随机初相、Hann 包络）；SNR 定义为注入 RMS 与信号 RMS 之比；表中为 5 名被试的准确率均值。

按“run-balanced recall 超过真实基线 + 2 倍标准差”定义检测阈：频带功率约 **SNR 0.5**，Hjorth 约 0.8，ERP 对 alpha 注入不敏感。真实数据校正后约为 9.6%，对应 SNR≈0，说明其语义效应若存在，幅度低于 0.3–0.5 的注入信号（图 6）。

![图 6　半合成 alpha 爆发注入下的准确率—SNR 曲线（左：原始特征；右：位置去趋势）](../figures/injection_curve.png)

### 4.8 讨论

**为什么此前结果偏高。** 随机划分让同一 run 的相邻试次进入训练与测试；类别按块呈现使位置成为强代理；类别间句长分布不同又提供了第二条捷径；跨任务配对评估则共享漂移。四者在常规协议下叠加，足以把机会水平推到 12%–19%。

**对 BCI 的含义。** 单试次想象语音的语义信噪比可能远低于文献报告值。若要在真实 BCI 中利用语义信息，需要重复呈现与试次平均、随机化呈现顺序、或引入更强的锁时范式（如视觉/听觉提示）。仅靠更换更大的网络，难以突破检测阈。

**方法适用性。** 本文的“位置基线 + 位置去趋势 + run-balanced recall + 训练级置换 + 半合成注入”并不依赖特定模型，可直接用于其他按块呈现的 EEG 范式（如情绪、音乐、视频诱发的语义分类），作为论文投稿前的自检流程。

---

## 5 结论

本文系统检查了 Chisco 想象语音数据中语义解码的可信性。常规按 run 分组交叉验证可得到 13.1% 的 10 类准确率，但仅用 run 内试次位置即可达到 35.4%；位置去趋势后所有模型回落到 8.9%–10.4% 的 run-balanced recall，训练级置换显示打乱标签后准确率反而更高（11.8%）。半合成注入给出频带功率流程约 0.5 的 SNR 检测阈，而真实数据对应 SNR≈0。

因此，在该数据集与当前特征/模型下，没有可靠证据表明单试次 EEG 中含有可跨 run 解码的语义类别信息；此前报告的高准确率主要可由呈现顺序与评估协议解释。所提框架可作为 EEG 语义解码研究的混杂诊断与可信评估工具。

**局限**：仅 5 名被试、单一数据集；位置去趋势假设位置效应可由低阶多项式描述；注入信号为 alpha 爆发，对其它类型语义效应（如 ERP 幅值、相位耦合）的灵敏度可能不同；未做源定位、相位耦合或跨被试表征相似性分析，这些仍可能揭示更细的语义结构。

---

## 参考文献

[1] ZHANG Z H, DING X, et al. Chisco: An EEG-based BCI dataset for decoding of imagined speech[J]. Scientific Data, 2024, 11. DOI: 10.1038/s41597-024-04114-1.

[2] NAMDARI S G, IACOMI F, BARBIERI R, et al. Semantic alignment of EEG and text: a contrastive learning framework for decoding imagined speech[EB/OL]. arXiv:2610.20766, 2026.

[3] ZHANG Z H, BAO Y, et al. Is EEG-to-text feasible in real-world scenarios? An in-depth analysis using a neuropsychology-inspired benchmark[C]//Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics. 2026. DOI: 10.18653/v1/2026.acl-long.61.

[4] KAUFMAN S, ROSSET S, PERLICH C, et al. Leakage in data mining: formulation, detection, and avoidance[J]. ACM Transactions on Knowledge Discovery from Data, 2012, 6(4): 1-21.

[5] VAROQUAUX G, CHEPLYGINA V. Machine learning for medical imaging: methodological failures and recommendations for the future[J]. npj Digital Medicine, 2022, 5: 48.

[6] LAWHERN V J, SOLON A J, WAYTOWICH N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of Neural Engineering, 2018, 15(5): 056013.

[7] SCHIRRMEISTER R T, SPRINGENBERG J T, FIEDERER L D J, et al. Deep learning with convolutional neural networks for EEG decoding and visualization[J]. Human Brain Mapping, 2017, 38(11): 5391-5420.

[8] GRAMFORT A, LUESSI M, LARSON E, et al. MEG and EEG data analysis with MNE-Python[J]. Frontiers in Neuroscience, 2013, 7: 267.

[9] PEDREGOSA F, VAROQUAUX G, GRAMFORT A, et al. Scikit-learn: machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.

[10] BLANKERTZ B, LEMM S, TREDER M, et al. Single-trial analysis and classification of ERP components—a tutorial[J]. NeuroImage, 2011, 56(2): 814-825.

---

> **说明**：本文件为初稿，作者、单位、基金、全部图的最终版本（建议重绘为中文坐标轴）以及参考文献卷期/页码需在投稿前核对与补齐。所有数字来自 `plan_auto` 下可复现实验（见 `REPORT.md`）。
