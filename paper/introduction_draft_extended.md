# 1 引言（详细版）

想象语音（imagined speech）是指个体在不产生可听发声、也无明显构音动作的条件下，在脑内完成的言语表达过程。以想象语音为控制信号的脑机接口（brain–computer interface, BCI）有望为肌萎缩侧索硬化、脑卒中等导致严重言语障碍的患者提供一条不依赖外周肌肉的交流通道，因而被视为言语 BCI 中最具临床价值、同时也最具挑战性的范式之一<sup>[2]</sup>。在各类记录手段中，脑电（electroencephalography, EEG）因无创、便携、时间分辨率高而成为非植入式想象语音解码的主要选择<sup>[1,3]</sup>。然而，想象语音既没有声学输出可供对齐，也缺乏运动想象那样明确的外周参照，单试次信噪比低、个体差异大，其语义层面的解码长期局限于小词表与高度受控的实验条件<sup>[1,2]</sup>。

数据规模与质量是制约该方向的关键因素。已有中文想象语音研究多基于数十至数百试次的小样本，类别数有限，且多在单一采集时段内完成，难以支撑语义类别层面的统计推断。Chisco 中文想象语音脑电数据集的公开发布改变了这一状况：该数据集记录 5 名健康被试、每任务 45 个 run，包含两万余句中文句子，并为每句标注 39 个语义类别，是目前单人记录规模最大的想象语音 EEG 资源之一<sup>[4]</sup>。数据集同时提供了基于 PREP、Autoreject 与独立成分分析的标准化预处理结果<sup>[5,6]</sup>，为在统一数据与统一预处理下开展方法学比较创造了条件。

数据规模的扩大并不自动带来可信的结论，评估协议随之成为关键。EEG 具有显著的 session 与 run 级非平稳性：电极阻抗、头模佩戴位置、皮肤状态、疲劳程度以及残余眼电与肌电等，都会在一次连续采集内保持相对稳定，而在不同 run、不同采集日之间发生漂移<sup>[7]</sup>。这意味着同一 run 内的试次在特征空间中天然聚集，形成可被分类器识别的"记录指纹"。当采用随机划分或随机交叉验证时，同一 run 的试次会同时出现在训练集与测试集中，分类器可以借助这种记录层面的相似性取得高分，而这种优势并不能迁移到新的 run 或新的采集日。若刺激类别在各 run 之间的分布又不均衡，run 身份与句类标签之间便形成混杂，随机划分下的高准确率就难以被解释为对语义信息的解码。这一问题在机器学习领域被称为数据泄漏，其形成机制、检测与规避已有系统讨论<sup>[8]</sup>，但在想象语音 EEG 的实证研究中尚未得到充分重视。

与评估协议紧密相关的是统计判断的严谨性。仅报告"高于理论机会水平"的准确率并不足以支持"存在可解码信息"的结论：当类别不平衡、样本量有限时，理论机会水平本身就可能偏离实际零分布<sup>[10]</sup>；同时，小样本下的交叉验证估计波动较大，点估计之外还应给出置信区间<sup>[9]</sup>。因此，在报告任何高于机会水平的解码结果之前，至少应当提供基于标签置换的经验零分布、以采集单元（run 或 session）为分组的交叉验证，以及效应量的区间估计。遗憾的是，这三项要求在现有想象语音文献中的执行并不一致，更缺少对"泄漏究竟贡献了多少准确率"的定量刻画。

针对上述问题，本文以 Chisco 数据集中的中文想象语音为对象，在一套刻意保持简单的线性基线上，系统评估随机划分所获得的句类解码优势在多大程度上来自 run/session 指纹与 run–句类混杂。具体而言：（1）以逐通道 5 个频段的 Welch 相对功率（122 通道 × 5 频段，共 610 维）为特征，以标准化后的收缩线性判别分析（linear discriminant analysis, LDA）为分类器，在 10 个高频语义类别上建立基线；（2）采用随机 8:2 划分与 5 折交叉验证给出惯常报告口径下的准确率，并以 2000 次标签置换构造经验零分布；（3）将标签替换为试次所属的 run，检验 run 身份本身的可解码性（run 身份探针）；（4）统计每个 run 内多数句类的占比，并计算"仅用 run 身份预测句类"的多数类基线，以量化 run 与句类之间的混杂程度；（5）采用留一 run（leave-one-run-out, LORO）与留一天（leave-one-day-out, LODO）协议重新评估句类解码，并与随机 5 折结果做逐 run、逐日的配对比较。全部实验另做频段与通道扫描，以检验结论对特征子集的稳健性。选择线性特征与线性分类器而非深度网络，是为了使"准确率来自何处"能够被清晰归因：若在最有利于暴露记录伪影的简单模型上已能观察到显著效应，更复杂的模型只会放大而非消除这一问题。

结果表明：第一，run 身份极强可解码，5 名被试的交叉验证准确率为 97.2%~99.9%，机会水平为 2.2%，置换检验 p 值均达到可计算下限，说明当前记录条件下存在远强于"轻微泄漏"的记录指纹；第二，run 与句类严重混杂，每个 run 内多数句类的占比平均为 31%，仅凭 run 身份预测句类即可达到约 32% 的准确率，是 10% 机会水平的三倍以上；第三，随机划分的优势大部分是 run 伪影，平均准确率由随机划分的 22.4% 降至 LORO 的 11.9%，绝对下降约 10.5 个百分点，若以超出机会水平的部分衡量，约 84% 的优势在留出 run 后消失，且 5 名被试的逐 run 配对差值全部为正（Wilcoxon 符号秩检验 p 介于 3.3×10⁻⁷ 与 3.5×10⁻⁶ 之间）；第四，跨 run 的残余信息很弱、跨天基本没有，LORO 下仅 3 名被试显著高于机会水平，LODO 平均准确率降至 9.5%，5 名被试均不显著（p 为 0.52~0.998），提示 LORO 中残余的优势更可能来自同一采集日内的 session 相似性，而非稳定的句义编码。

本文的贡献有三。其一，在方法层面，构建并验证了一套可直接复用的诊断组合——run 身份探针、run–句类混杂量化，以及留一协议下的逐组配对比较——用于识别想象语音解码中的记录指纹与数据泄漏；其二，在实证层面，系统给出随机划分对句类解码准确率的高估幅度（约 8~11 个百分点，约 84% 的超机会优势），并说明该高估主要来自 run 层面而非日层面；其三，在规范层面，建议想象语音 EEG 研究在报告任何高于机会水平的解码结果时，至少同时提供分组交叉验证与置换零分布，并明确说明刺激类别在各采集单元间的分布是否均衡。本文其余部分安排如下：第 2 节介绍数据与预处理，第 3 节说明标签构造与评估协议，第 4 节给出实验结果，第 5 节讨论其含义与局限。

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

1. **数字口径与来源**：22.4%（随机划分参考 CV）、11.9%（LORO）、9.5%（LODO）、98.4%（run 身份）、32%（run 多数类基线）、31%（run 内多数类占比）均取自
   `outputs/experiment_summary/summary_by_experiment.csv`；"约 84%" = 1 − (0.119 − 0.100)/(0.224 − 0.100)；
   逐被试 LORO 配对 p 值见 `outputs/loro_lodo/main_8_2/summary_all.csv`。若正文改用 8:2 的 test 准确率（23.9%）作为"随机划分"口径，则高估幅度改为约 12 个百分点，需全文保持一致。
2. **文献需核实**：第 [2][3][4][7][9][10] 条的卷期页码请按数据库核对；[4] 的作者名单建议补全（Zi-Han Zhang, Xiao Ding, …, Ting Liu）。
3. **"执行并不一致"一句**：审稿人可能要求具体文献佐证。建议补 2~3 篇仅用随机划分报告想象语音解码的实证论文，或改成不评价他人做法的中性表述（如"在已报道的结果中，随机划分仍是常见口径"）。
4. **压缩建议**：若期刊限制引言篇幅，可合并第 3、4 段，删去第 2 段中关于预处理工具的细节（PREP、Autoreject），并保留四段式：背景—问题—方法—结果与贡献。已另存压缩版 `paper/introduction_draft.md`。
5. **期刊格式**：多数中文期刊的引言不编号，可将标题改为"引言"并去掉"1"；参考文献请按目标期刊的 GB/T 7714 版本调整（如是否保留 DOI、作者是否全列）。
6. **与前文衔接**：第 2 节可直接对应 `paper/dataset_and_preprocessing.md`，第 3 节对应 `paper/methods_draft.md`；后者 2.4 分类模型仍为待补，引言中的特征与分类器描述需与其保持一致。
7. **摘要与关键词**：尚未撰写，建议在引言定稿后提炼"想象语音、脑电、数据泄漏、留一 run、语义解码"等关键词。
