# NRA-OPSD 算法说明：anchor-local 四格与可恢复投影

本文先用中文解释符号和四格构造，再给出与纯张量实现一致的公式。这里的“证据”是在显式匹配、cross-fitting 和 permutation calibration 条件下的关联证据，不是无条件因果结论。当前仓库尚未实现真实数据 runner/model trainer；以下真实数据 protocol 是待执行的研究设计，synthetic 仅验证代码 contract。

## 1. 符号先翻成试次

| 符号 | 中文含义 | 必须满足的条件 |
|---|---|---|
| `i` | 当前锚点（anchor）试次 | 所有比较围绕它构造 |
| `x_i` | 锚点试次 `i` 的脑记录 | student 推理时真正可见的输入 |
| `p_i` | `x_i` 对应的真实发音/音素序列 | `x_i` 与 `p_i` 是四格中唯一匹配的一格 |
| `j` | 负发音的索引 | 不是把另一个 transcript 当作正确答案 |
| `p_j` | 从同一 split/fold/subject/session/`reliability_stratum` 匹配池的 trial 采样的负发音 | 必须与 `p_i` 不同；confirmatory 中各 `p_j` 还要彼此不同 |
| `c` | 单独的控制试次 | 与 anchor 同 split/fold/subject/session/`reliability_stratum`，并满足时长约束 |
| `x_c` | 控制试次 `c` 的脑记录 | 它自己的真实发音是 `p_c` |
| `p_c` | `x_c` 真正对应的发音 | 必须同时不同于 `p_i` 和 `p_j` |

例如，real loader 先在官方 split 内建池；`x_i` 来自其中一个 fold、受试者 session 和 reliability stratum 中说“猫”（示意音素 `/mao/`）的试次，所以 `p_i=/mao/`。从同一 split/fold/subject/session/stratum 选“门”（`p_j=/men/`）作负发音，再选实际说“灯”（`p_c=/deng/`）的控制脑记录 `x_c`。于是只有“`x_i` + `/mao/`”真实匹配，其他三格都不匹配。

有些评估公式把 matched-wrong brain 记为 `x_j`。`x_j` 是“任务指标的错误脑反事实”记号；四格估计中的 `x_c` 是有额外 `p_c ∉ {p_i,p_j}` 约束的控制脑。两者不能因为下标相似而自动当作同一个样本。

## 2. 为什么是 anchor-local 四格

四次 teacher forward 必须共享完全相同的 student-generated prefix、token alignment 和 vocabulary support：

| 脑输入 \ 特权发音 | `p_i`（锚点真发音） | `p_j`（负发音） |
|---|---|---|
| `x_i`（锚点脑） | **匹配** `(x_i,p_i)` | 不匹配 `(x_i,p_j)` |
| `x_c`，且 `p_c ∉ {p_i,p_j}` | 不匹配 `(x_c,p_i)` | 不匹配 `(x_c,p_j)` |

令 `z_t(x,p)` 是 detached teacher 在时间步 `t` 的词表 log-probability。以当前 detached student 分布 `s_t` 做 Fisher centering：

```math
C_{s_t}[a] = a-\left(\sum_v s_t(v)a(v)\right)\mathbf 1,
\qquad
\langle a,b\rangle_{s_t}=\sum_v s_t(v)a(v)b(v).
```

anchor-local difference-in-differences 为：

```math
\Gamma_{icj}^t
=C_{s_t}\!\left[
\{z_t(x_i,p_i)-z_t(x_i,p_j)\}
-\{z_t(x_c,p_i)-z_t(x_c,p_j)\}
\right].
```

第一行问“对 anchor brain，真发音相对负发音改变了什么”；第二行减去“对一个自身发音既不是 `p_i` 也不是 `p_j` 的控制 brain，同样的发音替换改变了什么”。如果 teacher 只是与 brain 无关地复制发音，两行的加性发音效应相同并被消掉。

四格并不能自动消除 brain-dependent gain、attention saturation、SNR 或 prefix compatibility。因而 `Γ` 仍不能单独作为可转移证据，后面还要经过 exact-statistic permutation null、frozen cross-fitted brain-only support、reliability 和 cross-session stability。

## 3. 为什么 `K>1`，而且负发音必须互不相同

一个 `(x_c,p_j)` 组合可能因为偶然音素相似、长度边界、某个控制试次的噪声或 decoder prefix compatibility 而产生异常方向。只用一个负例会让 target 依赖某次抽样。对同一 anchor 采样 `K` 个合格 `(c,j)` 组合后聚合：

```math
\bar\Gamma_i^t
=C_{s_t}\!\left[\sum_{(c,j)=1}^{K}w_{icj}\Gamma_{icj}^t\right],
\qquad w_{icj}\ge 0,\quad \sum w_{icj}=1.
```

默认是 uniform mean；componentwise median 只作为预注册 robustness ablation。聚合后必须再次按同一个 `s_t` 居中。

`p_j` 互不相同很重要：重复四次同一发音只是给同一个对比重复计权，不能提供四个独立的发音扰动，还会虚报 effective `K`。因此：

- synthetic smoke 使用 `K=2`，只为了低成本执行路径；
- confirmatory 使用至少四个 **不同的** `p_j`，当前配置为 `K=4`；
- real loader 先按官方 split 建池；每个 `p_j` 来源 trial 和 `x_c` 都与 anchor 同 split/fold/subject/session/`reliability_stratum`，并满足预注册时长容差；`p_c` 同时区别于对应 `p_i`、`p_j`；
- 合格候选不足时记录 shortfall/effective `K`；confirmatory 样本跳过，不能放松约束、跨 session 补样或重复发音凑数。

## 4. brain-only 支持与 exact null

从一个 frozen、cross-fitted、从未接收 privilege 的 CE checkpoint 计算脑对比：

```math
\bar B_i^t
=C_{s_t}\!\left[
\sum_c\omega_{ic}
\{\log p_S^t(\cdot\mid x_i)-\log p_S^t(\cdot\mid x_c)\}
\right].
```

只保留 `Γ̄` 在正向 brain-only 方向上的分量：

```math
u_i^t=
\frac{[\langle\bar\Gamma_i^t,\bar B_i^t\rangle_{s_t}]_+}
{\lVert\bar B_i^t\rVert_{s_t}^2+\epsilon}\bar B_i^t.
```

完整 signed normalized alignment 是：

```math
A_i^t=
\frac{\langle\bar\Gamma_i^t,\bar B_i^t\rangle_{s_t}}
{\lVert\bar\Gamma_i^t\rVert_{s_t}\lVert\bar B_i^t\rVert_{s_t}+\epsilon}.
```

在 training-only held-out fold 中置换发音分配，同时固定 prefix、stratum、`K`、sampler、aggregation、weights 与 vocabulary support。`τ` 是这个 **完整统计量** 的 `(1-α)` 分位数，而不是另一个近似 cosine 的阈值。真实 pilot 计划至少 99 次 permutation，confirmatory 至少 999 次，注册值是 `α=0.05`。转移强度为：

```math
\kappa_i^t=\operatorname{clip}
\left(\frac{A_i^t-\tau_i^t}{1-\tau_i^t+\epsilon},0,1\right).
```

实现中，calibration 不是可与任意 interaction 混用的裸 threshold。`NRAEvidence` 必须同时保存：frozen CE 的 `brain_only_direction`、`null_threshold`、`source="frozen_crossfit_ce"`、training split、producer fold/session/trial membership、当前 identity、checkpoint、`statistic_id="signed_fisher_alignment"`、生成该统计量所用的 `statistic_epsilon`、`alpha=0.05`、phase/permutation count、`K`、sampler、aggregation/weight policy、seed、paired prefix/alignment/support policy，以及 `control_policy="anchor_local_three_mismatch"`。producer membership 必须非空且排除当前 fold/session/trial；pilot 至少 99 permutations 且 `K≥1`，confirmatory 至少 999 且 `K≥4`。

当前 interaction 则封装为 `NRAInteractionBatch`，而不是裸 tensor。它把 `values`/可选 weights 与当前 subject、pronunciation、reliability stratum、fold/session/trial、checkpoint、prefix/alignment/support、sampler、aggregation/weight policy、seed 和 control policy 绑定。主调用形式为：

```python
target, diagnostics = recoverable_target(
    mode="nra",
    student_log_probs=student_log_probs,
    teacher_log_probs=teacher_log_probs,
    anchor_interactions=nra_interaction_batch,
    nra_evidence=nra_evidence,
    neural_reliability=neural_reliability,
    session_stability=session_stability,
    epsilon=nra_evidence.statistic_epsilon,
)
```

接口会逐字段验证 batch 与 evidence，另检查实际 negative axis 的 `K`、aggregation 和可验证的 uniform weights。调用时 `epsilon` 必须与 evidence 中生成 calibration statistic 的 `statistic_epsilon` 完全相等；任一 context/statistic mismatch 都 fail closed。NRA 模式拒绝裸 interaction tensor、单独的 `interaction_weights`，也拒绝任何 `null_interaction`（包括零向量）；raw-null subtraction 只保留在显式 `interaction_only_ablation`，不能作为主 null calibration。

### Synthetic favorable fixture 与真实 calibration 的边界

CPU smoke 使用与 current path 相同的 frozen synthetic teacher，但为 calibration 单独构造 3 个 held-out neural record（1 anchor + 2 controls）和 99 个 pronunciation-label donor。current 与 calibration 都标为 train；两 cohort 的 fold/session/trial ID 和 neural tensor storage 不相交，而各 cohort 内记录共享 subject、session 与 `reliability_stratum`。给定 seed 只决定全部 99 个 cyclic shift 的顺序，所以这 99 个 assignment 不是 99 个独立随机 permutations。

每个 assignment 用同一个 exact `signed_fisher_alignment` 统计量和登记的 `statistic_epsilon=1e-8`，synthetic threshold 是 `0.95` higher quantile；98 个 assignment 破坏 anchor pairing，3 个任一被评分格含 anchor label，1 个保持 anchor match。这个设计是刻意 favorable 的 deterministic smoke fixture，用来检查 target/statistic/provenance plumbing。它既不是由真实神经数据得到的 cross-fitted calibration，也不能支持 interaction-existence、benchmark 或统计显著性结论。

## 5. 只重建神经支持的 teacher correction

完整 privilege correction 为：

```math
r_i^t=C_{s_t}[z_t(x_i,p_i)-\log s_t].
```

再把它正向投影到 `u`，并乘上 reliability `ρ`、cross-session stability `σ` 和 null-excess `κ`：

```math
\beta_i^t=
\frac{[\langle r_i^t,u_i^t\rangle_{s_t}]_+}
{\lVert u_i^t\rVert_{s_t}^2+\epsilon},
\qquad
r_{NR,i}^t=\rho_i^t\sigma_i^t\kappa_i^t\beta_i^t u_i^t,
```

```math
\lambda_{clip,i}^t=
\begin{cases}
1,&\lVert r_{NR,i}^t\rVert_\infty=0,\\
\min\left(1,\dfrac{c}{\lVert r_{NR,i}^t\rVert_\infty}\right),&\text{otherwise},
\end{cases}
\qquad
q_{NR,i}^t=\operatorname{softmax}
\left(\operatorname{stopgrad}(\log s_t)
+\alpha_{step}\lambda_{clip,i}^t r_{NR,i}^t\right).
```

这里把 step size 写作 `α_step`，避免与 null test 的显著性水平 `α=0.05` 混淆。实现把 `lambda_clip` 作为 `clip_scale` 诊断；clipping 只乘一个整体 scalar，因此不会把 correction 旋出 brain-supported span。若 interaction 为零、与 brain-only 方向正交/反向、低于 null，或完整 teacher correction 不沿受支持方向，则 `q_NR` 回到 student 分布，不传递该 privilege correction。

诊断将两个 coverage 分开：`geometric_coverage` 要求 interaction/projection/`kappa` 的几何条件成立；`transfer_coverage` 在此基础上还要求 `ρ>0` 且 `σ>0`。旧名 `coverage` 是 `transfer_coverage` 的兼容 alias，而不是纯几何 coverage。

## 6. Tensor 边界与对比公平性

- teacher、frozen CE support、所有 interaction/weight/gate/null statistic 和 `q_NR` 都 detach；只有 live student 获得梯度。
- student rollout 与最终推理均不接收 `p_i`、transcript 或 phoneme privilege。
- NRA 的 typed evidence/batch identity 不完整、互相不匹配或无法验证 uniform weight 时直接拒绝，不静默退化为 interaction-only。
- NRA mode 要求显式 reliability/session-stability 输入；projection 会 detach 和范围校验，但它们仍是上游提供的裸 scalar/tensor，尚无端到端 typed gate artifact 或真实 orchestration。
- student log-probabilities 先在内部高精度工作 dtype 中计算 vocabulary-aware `logsumexp`，按 source dtype 与词表大小决定容差；只有通过归一化质量检查的输入才会在 Fisher 运算前减去 log-mass。large-vocabulary BF16 因而可用，但明显未归一化的输入仍被拒绝。
- `target_kl` 是数值诊断；其理论非负值若因浮点舍入略低于零，报告值会 clamp 到 `0`。
- reliability/null calibration 的 pronunciation ID 必须和当前 pronunciation 精确一致；validity mask 只接受 boolean 或有限二值 `{0,1}` tensor。
- stability bank 只含 training-only、checkpoint-matched、leave-current-trial/session-out 的 raw contrast；比较时双方用同一 `s_t` 重居中。
- primary/confirmatory 缺少跨 session 支持时 fail closed 为 stability `0`，不发生 transfer，并记录 missing-support flag；`1.0` 若保留，只能是显式命名的 ablation，不能把缺失伪装成完美稳定。
- 主比较固定 vocabulary support、forward/token budget、sampler、decoding、seeds 与 trainable-parameter budget。
- `legacy_dual_cosine` 仅是消融：它缩放整个 privileged KL；`interaction_projection` 才会移除神经不支持的 teacher correction 维度。

synthetic 的 CE arm 使用 gold prefix；各 OPSD arm 则从当前 brain-only student 生成 detached greedy rollout，并在该 prefix 上比较。legacy arm 只将 gate 乘到 vanilla arm 同一个未经重建的 privileged-teacher KL。真实 dataset runner、decoder/trainer 集成和真实 evaluator 尚不在当前实现内。

因此，上述 detach 性质证明的是现有纯 tensor constructor/objective 的局部梯度边界，不是对整个真实 pipeline 的 leakage-proof 声明。真实 runner 仍须验证数据流、checkpoint ownership 和 gate artifact provenance。
