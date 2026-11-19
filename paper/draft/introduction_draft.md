# 1 引言

想象语音（imagined speech）是指个体在不产生可听发声、也无明显构音动作的条件下，在脑内完成言语表达的过程。以想象语音为控制信号的脑机接口（brain–computer interface, BCI）有望为肌萎缩侧索硬化、脑卒中等导致严重言语障碍的患者提供一条不依赖外周肌肉的交流通道，因而被视为言语 BCI 中临床价值最高、同时难度最大的范式之一<sup>[2]</sup>。脑电（electroencephalography, EEG）因无创、便携、时间分辨率高而成为非植入式想象语音解码的主要手段<sup>[1,3]</sup>。但想象语音既没有声学输出可供对齐，也缺乏运动想象那样明确的外周参照，单试次信噪比低、个体差异大，其语义层面的解码长期局限于小词表与高度受控的实验条件<sup>[1,2]</sup>。

数据规模是制约该方向的重要因素。Chisco 中文想象语音脑电数据集的公开发布改变了这一状况：该数据集记录 5 名健康被试、每任务 45 个 run，包含两万余句中文句子，并为每句标注 39 个语义类别，是目前单人记录规模最大的想象语音 EEG 资源之一<sup>[4]</sup>；其标准化预处理结果<sup>[5,6]</sup>也为在统一数据上开展方法学比较创造了条件。

然而，数据规模的扩大并不自动带来可信的结论，评估协议随之成为关键。EEG 具有显著的 session 与 run 级非平稳性：电极阻抗、头模佩戴位置、皮肤状态与疲劳程度等在一次连续采集内保持相对稳定，而在不同 run、不同采集日之间发生漂移<sup>[7]</sup>。因此同一 run 的试次在特征空间中天然聚集，形成可被分类器识别的"记录指纹"。当采用随机划分或随机交叉验证时，同一 run 的试次会同时出现在训练集与测试集中，分类器可借助记录层面的相似性取得高分，而这种优势无法迁移到新的 run 或采集日；若刺激类别在各 run 间分布不均，run 身份与句类标签还会形成混杂，随机划分下的高准确率便难以解释为对语义信息的解码。该现象在机器学习中被称为数据泄漏，其形成机制与规避已有系统讨论<sup>[8]</sup>，但在想象语音 EEG 的实证研究中尚未得到充分重视。与之相伴的还有统计判断的不足：仅"高于理论机会水平"并不足以支持"存在可解码信息"，还需以采集单元为单位的分组交叉验证，以及基于标签置换的经验零分布<sup>[9,10]</sup>。

针对上述问题，本文以 Chisco 数据集中的中文想象语音为对象，在一套刻意保持简单的线性基线上，系统评估随机划分所获得的句类解码优势在多大程度上来自 run/session 指纹与 run–句类混杂。研究采用逐通道多频段 Welch 功率特征与收缩线性判别分析，在同一数据、同一特征与同一分类器上分别实施随机划分、run 身份探针、留一 run 与留一天协议，并通过逐组配对比较与标签置换检验判断差异的可靠性。选择线性模型而非深度网络，是为了使"准确率来自何处"能够被清晰归因：若最有利于暴露记录伪影的简单模型已呈现明显效应，更复杂的模型只会放大而非消除这一问题。

本文的贡献有三：其一，构建并验证一套可直接复用的诊断组合——run 身份探针、run–句类混杂量化与留一协议下的逐组配对比较——用于识别想象语音解码中的记录指纹与数据泄漏；其二，定量刻画随机划分对句类解码准确率的高估幅度，并区分该高估来自 run 层面还是采集日层面；其三，据此建议想象语音 EEG 研究在报告高于机会水平的解码结果时，同时提供分组交叉验证与置换零分布，并说明刺激类别在各采集单元间的分布是否均衡。本文其余部分安排如下：第 2 节介绍数据与预处理，第 3 节说明标签构造与评估协议，第 4 节给出实验结果，第 5 节讨论其含义与局限。

## 参考文献

[1] LOTTE F, BOUGRAIN L, CICHOCKI A, et al. A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update[J]. Journal of Neural Engineering, 2018, 15(3): 031005.

[2] MARTIN S, BRUNNER P, HOLDGRAF C, et al. Decoding inner speech using electrocorticography: progress and challenges toward a speech prosthesis[J]. Frontiers in Neuroscience, 2014, 8: 103.

[3] DÉFOSSEZ A, CAUCHETEUX C, RAPIN J, et al. Decoding speech perception from non-invasive brain recordings[J]. Nature Machine Intelligence, 2023, 5(10): 1097-1107.

[4] ZHANG Z H, DING X, et al. Chisco: an EEG-based BCI dataset for decoding of imagined speech[J]. Scientific Data, 2024, 11: 1265.

[5] BIGDELY-SHAMLO N, MULLEN T, KOTHE C, et al. The PREP pipeline: standardized preprocessing for large-scale EEG analysis[J]. Frontiers in Neuroinformatics, 2015, 9: 16.

[6] JAS M, ENGEMANN D A, BEKHTI Y, et al. Autoreject: automated artifact rejection for MEG and EEG data[J]. NeuroImage, 2017, 159: 417-429.

[7] SAHA S, BAUMERT M. Intra- and inter-subject variability in EEG-based sensorimotor brain computer interface: a review[J]. Frontiers in Computational Neuroscience, 2020, 13: 87.

[8] KAUFMAN S, ROSSET S, PERLICH C. Leakage in data mining: formulation, detection, and avoidance[J]. ACM Transactions on Knowledge Discovery from Data, 2012, 6(4): 15.

[9] VAROQUAUX G. Cross-validation failure: small sample sizes lead to large error bars[J]. NeuroImage, 2018, 180: 68-77.

[10] COMBRISSON E, JERBI K. Exceeding chance level by chance: the caveat of theoretical chance levels in brain signal classification and statistical assessment of decoding accuracy[J]. Journal of Neuroscience Methods, 2015, 250: 126-136.

---

## 写作说明（非正文，定稿前删除）

1. **本版为压缩版**，正文约 1150 字（不含标点），只保留背景与缺口、研究目标、三条贡献；具体特征维度、划分比例、置换次数与全部结果数字已移出，待写入第 3 节（方法）与第 4 节（结果）。
2. **原扩展版可回收的内容**：run 身份 97.2%~99.9%、随机划分 22.4%→LORO 11.9%、约 84% 优势消失、LODO 9.5% 不显著等数字，建议放在结果节首段作总览，不要在引言中出现。
3. **文献需核实**：第 [2][3][4][7][9][10] 条的卷期页码请按数据库核对；[4] 作者名单建议补全（Zi-Han Zhang, Xiao Ding, …, Ting Liu）。
4. **如需更短**（800 字以内）：可合并第 2、3 段，删去 Chisco 的句子数与类别数，并删去"选择线性模型而非深度网络"的归因说明，仅保留三条贡献。
5. **期刊格式**：多数中文期刊引言不编号，可将标题改为"引言"；参考文献按目标期刊的 GB/T 7714 版本调整（是否保留 DOI、作者是否全列）。
6. **衔接**：第 2 节对应 `paper/draft/methods_draft.md` 的 2.1–2.3，第 3 节对应其 2.4–2.5（其中 2.4 分类模型仍待补）。
