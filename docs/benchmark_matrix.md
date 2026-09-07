# 基准矩阵：数据集、基线、指标、控制与停止门

本页是 **planned comparison matrix**，不是已实现方法清单或结果表。配置中的 `SUPPORTED_METHODS` 只登记 task-aware resolver 可接受的方法名，不代表相应 trainer、runner 或 evaluator 已实现。B2T24、B2T25 与 BrainHub/UMBRAE 真实运行全部为 **NOT RUN**；所有数值都是实验设置或门限，没有真实得分。synthetic smoke 也不是 benchmark result，本文不作 SOTA 声明。当前运行状态见 [experiment_status.md](experiment_status.md)。

## 1. 三套真实数据主矩阵

| Track | 固定配置 / adapter | 主方法与 privilege | 主指标 | 次指标与诊断 | brain-dependence 指标 | Seeds / `K` / null | 资源门 | 当前状态 |
|---|---|---|---|---|---|---|---|---|
| Brain-to-Text 2024 | `configs/b2t24/nra_opsd.yaml`; BIT | `interaction_projection`; phonemes | official aggregate WER（越低越好） | S/D/I、CER、exact match、truncation、per-session、corpus-type；KL、entropy、generation length、coverage/stability | `WER(wrong)-WER(real)` 的等价“real quality minus wrong quality”方向，并报告相对 CE 的 difference-in-gap | `41/42/43`; `K=4` 个 distinct pronunciations；999 permutations，`α=0.05` | 16 GiB profile；peak allocation ≤ 14.5 GiB；估计 14.25 GiB | **NOT RUN** |
| Brain-to-Text 2025 | `configs/b2t25/nra_opsd.yaml`; BIT | `interaction_projection`; phonemes | official aggregate WER（越低越好） | 与 B2T24 相同，另按官方 split/session/corpus 报告 | 与 B2T24 相同；相对同数据 CE baseline 的 difference-in-gap | `41/42/43`; `K=4`; 999，`α=0.05` | 16 GiB profile；peak allocation ≤ 14.5 GiB；估计 14.35 GiB | **NOT RUN** |
| BrainHub grounding / UMBRAE | `configs/brainhub/umbrae_grounding.yaml`; BrainHub manifest contract，计划 UMBRAE backend | `evidence_iou_rlvr`; image/grounding evidence；adapter/projector/LoRA 是未实现计划 | original BrainHub protocol 的 held-out grounding IoU | IoU≥0.5 rate、invalid-output rate；all-instance/absent-object corrected protocol；多实例 Hungarian matching 的 FP/FN；reward zero-std skip rate | `U_ground = IoU(real,b*) - IoU(matched-wrong,b*)`；并相对 coordinate-token CE 报告 difference-in-gap | `41/42/43`; confirmatory control `K=4`; 999，`α=0.05` | 24 GiB profile；peak allocation ≤ 22 GiB；估计 21.5 GiB | **NOT RUN** |

WER 越低越好，所以表中 brain gap 使用 `WER(wrong)-WER(real)`，与统一定义 `quality(real)-quality(wrong)` 同向：正值才表示正确脑更好。不得把符号相反的 raw WER 差混入同一列。

UMBRAE reward 固定为：

```math
r_{box}=\operatorname{IoU}(\hat b,b^*)
+0.25\,\mathbb 1[\operatorname{IoU}\ge0.5]
-1.0\,\mathbb 1[\text{invalid}].
```

threshold bonus 不能替代 dense IoU；matched-wrong trial 首先是独立诊断，不得默认塞进 reward 来制造更大的 gap。

当前 task-aware resolver 会校验 visual-grounding RLVR 配置是否精确提供有限的 `iou_threshold`、`threshold_bonus` 与 `invalid_penalty`，以及各自范围；这只是 config contract，不表示 grounding trainer 或 evaluator 已实现。

## 2. Brain-to-Text 基线矩阵

所有 matched comparisons 使用同一数据边界、trainable parameters、decoder/prefix policy、decoding、vocabulary support、forward/token budget、negative sampler 和 seeds。

| ID | 基线/方法 | 它检验什么 | 比较状态要求 |
|---|---|---|---|
| B1 | BIT CE + base contrastive alignment | 官方可复现 warm start 与 brain-conditioned reference | P2 必须先复现；所有 gap 的 reference |
| B2 | Scheduled sampling | 只处理 gold-prefix / rollout-prefix mismatch | 与 CE 匹配计算量 |
| B3 | Off-policy privileged KD on gold prefixes | privilege distillation，但不在 student rollout prefix 上 | 与 OPSD 区分 prefix 效应 |
| B4 | Minimum-WER / MBR | 直接序列目标，无 NRA target | WER training 必须至少与其比较 |
| B5 | Vanilla full-target OPSD / `uniform` | 不做 evidence projection 的主 OPSD baseline | 六臂 pilot 的 uniform arm |
| B6 | Vanilla WER-RLVR | 可验证 WER reward，但无 NRA target | true reward 与 random reward 匹配计算量 |
| B7 | `teacher_contrast` | 只用 privileged teacher contrast | 六臂 pilot |
| B8 | `brain_contrast` | 只用 brain contrast | 六臂 pilot |
| B9 | `reliability` | reliability-weighted full-target OPSD | 六臂 pilot |
| B10 | `legacy_dual_cosine` | scalar gate，仍缩放整个 privileged KL | 六臂 pilot，仅作消融 |
| B11 | `interaction_projection` / NRA-OPSD | anchor-local interaction + Fisher projection | 主方法；六臂 pilot 与 confirmatory |
| B12 | NRA-OPSD then WER-RLVR | 分阶段目标；不把 OPSD/RLVR 偷偷混成一个 scalar | 仅在 NRA-OPSD gate 通过后 |
| R1 | Published CTC / n-gram cascade | benchmark reference | 明确标为非 matched end-to-end reference |

Transcript privilege 只能作为 oracle upper-bound arm；phoneme privilege 是主条件。不得把 CTC cascade 叫作 OPSD，也不得假设 7B 必然优于 compact audio-LLM；7B 是 scale control。

当前 `configs/experiments/ablation.yaml` 精确注册六个**计划** arm：`uniform`、`reliability`、`teacher_contrast`、`brain_contrast`、`legacy_dual_cosine`、`interaction_projection`，seeds `41/42/43`，`K=2`，duration tolerance `0.15`，mean aggregation，单 rollout，max generation tokens `128`。task-aware resolver 会检查方法 family；仓库尚无执行这六臂真实训练的 trainer。这是 mechanism pilot 配置契约，不是 full confirmatory 或 pilot result。

## 3. BrainHub/UMBRAE 基线矩阵

| ID | 基线/方法 | 必报比较 |
|---|---|---|
| G1 | Published image-feature MSE / denoising alignment | 固定 checkpoint 的 fidelity reference |
| G2 | Coordinate-token CE | 主 CE baseline 与 difference-in-gap reference |
| G3 | Smooth-L1 + GIoU box head | 非生成式几何目标 baseline |
| G4 | Vanilla IoU policy optimization | 与 evidence-aware RLVR 匹配 reward budget |
| G5 | Evidence-aware IoU policy optimization | 当前 confirmatory config 的方法族 |
| G6 | Image-privileged vanilla OPSD | privilege、有 projection 无/否的对照 |
| G7 | Image-privileged evidence-gated OPSD | grounding 上的 evidence gating 对照 |

原始 BrainHub protocol 和 all-instance/absent-object corrected protocol 必须并列。多实例场景用 Hungarian matching，并显式计算 false-positive / false-negative penalty。只有 training verifier 改善而 held-out official metric 不改善时，方法不通过。

## 4. 每张主表不可省略的 falsification controls

| 维度 | 必须包含 | 判读 |
|---|---|---|
| 脑输入 | correct brain、zero brain、channel/time shuffled brain、same-split/fold/subject/session/`reliability_stratum` matched-wrong trial | 正确脑相对错误/破坏脑必须产生可解释的 held-out gap |
| Privilege | correct privilege、shuffled privilege | 排除纯 privilege copier |
| Reward | true reward、random reward，matched compute | true reward 不优于 random reward 即失败 |
| Model family | 技术上可行时至少一个 non-Qwen decoder | 单一家族效应必须限定结论 |
| 重复性 | 至少三个 distinct seeds；confirmatory 为 `41/42/43` | 报告逐 seed 与配对区间，不只报最佳 seed |
| Sampling | real loader 先按官方 split 建池；matched negative、`p_j` 来源 trial 与四格 control 均同 split/fold/subject/session/`reliability_stratum`；duration/length tolerance、distinct `p_j`、`p_c` 区别于 anchor/negative、effective-`K`/shortfall | 不得跨约束补样；confirmatory shortfall 样本跳过并记录 |
| Geometry | 同 prefix/alignment/vocabulary support；shuffled-phoneme 与 shuffled-brain null | raw uncentered Euclidean cosine 不是主统计量 |
| Provenance | split/fold/subject/session/trial/pronunciation/checkpoint/prefix/alignment/support/seed/permutation count | calibration、reliability、stability 均 training-only、cross-fitted、leave-current-out |
| Typed NRA boundary | `NRAEvidence` + `NRAInteractionBatch` 的 exact context/statistic identity；`statistic_epsilon` 与调用 epsilon 完全一致；实际 `K`/aggregation/uniform weights 可验证 | mismatch、epsilon drift、裸 tensor、separate weights 或任何 NRA `null_interaction` 都 fail closed；raw null 仅用于显式 ablation |

附加诊断包括 target KL、entropy、generation length、invalid-output rate、reward-group zero-standard-deviation skip rate、projection magnitude、`geometric_coverage`、`transfer_coverage`（兼容 alias `coverage`）、reliability、stability、negative validity 和 per-session performance。几何 coverage 不能冒充 reliability/stability gate 后的实际 transfer coverage；`target_kl` 会把仅由数值舍入造成的微小负值截到零。

## 5. 顺序执行与 stop/go gates

| Gate | 最小输入 | GO 条件 | STOP / 不得声明的情况 |
|---|---|---|---|
| P0 CPU synthetic CI | unit tests、toy overfit、checkpoint-resume equivalence；CE、vanilla OPSD、legacy、NRA、reward 路径 | finite loss；只有 student 有梯度；detachment/机制不变量通过 | 任一路径非有限、target 泄漏梯度，或 synthetic invariant 失败。即使 GO，也只允许说“代码路径通过 smoke” |
| P1 real-shape smoke | 每个 adapter 8 个真实形状样本、3 个 training steps | shape/gradient/finite loss/target detach 通过，且 peak memory 不越 14.5/22 GiB 门 | OOM、非有限、私有输入契约错误或超资源门 |
| P2 baseline fidelity | BIT validation behavior；一个 pinned UMBRAE/BrainHub checkpoint | 在注册容差内复现各自 baseline/evaluator 行为后冻结 reference | 未复现 baseline 就不得测试或宣传新 objective |
| P3 interaction-existence pilot | matched `Γ`，shuffled-brain / shuffled-phoneme exact-statistic null，cross-session bank | held-out interaction magnitude 高于注册 null interval 且跨 session 稳定/coverage 可接受 | interaction 不高于 null、跨 session 不稳定或无 cross-session support：primary transfer 为零并停止 NRA 主张，先检查匹配/测量 |
| P4 mechanism pilot | 六臂 matched-compute ablation；held-out real-vs-wrong benefit | NRA target 比 simpler gates 更能预测/产生 held-out brain benefit | NRA target 不优于 simpler gates：不进入 full confirmatory |
| P5 full evaluation | B2T seeds `41/42/43`、`K≥4`、paired bootstrap；BrainHub unique-image cluster bootstrap | 官方 held-out task metric与 brain gap 同时支持，经所有 falsification checks | 仅 task metric 改善、仅 verifier 改善、或 brain gap 不超过 CE 都不支持“更好的 brain decoding” |

跨 gate 的硬停止条件：

- true reward 不优于 random reward（matched compute）；
- real-versus-wrong brain gap 不比 CE 改善；
- WER training 不优于 MWER/MBR；
- grounding 不优于 GIoU 或 matched IoU-RL baseline；
- 只有训练 verifier 改善，official held-out metric 不改善；
- 效应只存在于一个 model family 且结论未作限定。

这些 gate 目前只作为预注册执行顺序。P0 只有 synthetic method smoke 与 contract tests，toy overfit/checkpoint-resume equivalence 尚未实现，所以仍是 **PARTIAL**。真实 dataset runner、BIT/UMBRAE model wiring、7B/其他 decoder、trainer、multi-instance Hungarian evaluator 和 paired/cluster bootstrap inference 尚未实现，因此 P1–P5 全部仍为 **NOT RUN**。
