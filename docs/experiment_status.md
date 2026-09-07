# 实验状态与声明边界

**状态快照：2026-09-07。** 当前交付是 research skeleton + executable contracts，不是 real end-to-end pipeline。本页区分 executable verification、配置预检和真实研究结果。B2T24、B2T25 与 BrainHub/UMBRAE 真实运行全部为 **NOT RUN**。没有 WER/IoU 结果，也没有推断或借用外部 leaderboard 数字；本仓库不作 SOTA 声明。

## 状态表

| 范围 | 状态 | 已能验证什么 | 明确不能声明什么 |
|---|---|---|---|
| Unit tests | 本地实现验证项；以当前 `pytest` 输出为准 | Fisher geometry、四格 interaction、projection/detachment、same-stratum matching、reliability pronunciation/mask contract、OPSD/reward/metrics、task-aware config/grounding-reward schema | 不能证明真实神经数据有效 |
| Synthetic CPU suite | **smoke only**；以 `python scripts/smoke_test.py` 或 `brain-train --synthetic ...` 的新鲜输出为准 | CE、vanilla OPSD、legacy dual-cosine、NRA-OPSD、RLVR 玩具代码路径；detached brain-only greedy rollout、finite loss、student-only gradient、determinism、target invariants | favorable synthetic null 不是真实 cross-fit neural calibration；不是 B2T/BrainHub metric 或 benchmark result；不能比较模型质量、SOTA 或统计显著性 |
| Typed NRA API | 本地 contract tests；不是数据实验 | `NRAEvidence` 与 `NRAInteractionBatch` exact context/statistic binding，包括 calibration/use `statistic_epsilon` 完全一致；显式 reliability/stability；span-preserving scalar clip；large-vocabulary BF16 student log-probability normalization；非负 `target_kl` 诊断；区分 geometric/transfer coverage；missing session support 不转移 | reliability/stability 仍以裸 detached values 传入；没有端到端 typed gate artifact，不能证明整个真实 pipeline leakage-proof 或真实 interaction 有效 |
| Config registry | resolver/dry-run contract | benchmark、task、adapter 与 method family 的兼容性校验；`brain-train`/`brain-evaluate` config dry-run 不访问私有数据 | `SUPPORTED_METHODS` 不表示对应 trainer 已实现；dry-run 不是训练或评估 |
| CLI execution | synthetic train smoke + config dry-run | synthetic 只由 `brain-train --synthetic` 执行；`brain-evaluate` 只解析 config dry-run | 没有真实 train/evaluate runner |
| B2T24 config | dry-run 可验证；真实实验 **NOT RUN** | YAML inheritance、confirmatory guards、BIT manifest contract、14.5 GiB 配置门 | 没有 real WER、brain gap、baseline fidelity 或显著性结果 |
| B2T25 config | dry-run 可验证；真实实验 **NOT RUN** | 与 B2T24 同类的配置/manifest contract | 没有 real WER、brain gap、baseline fidelity 或显著性结果 |
| BrainHub/UMBRAE config | dry-run 可验证；真实实验 **NOT RUN** | UMBRAE backend manifest contract、IoU reward config、22 GiB 配置门 | 没有 real IoU、IoU@0.5、`U_ground`、checkpoint fidelity 或显著性结果 |
| P0 CPU synthetic CI | **PARTIAL** | synthetic method smoke、detachment 与 contract tests 可执行 | toy overfit 与 checkpoint-resume equivalence 尚未实现，不能把整个 P0 标为 GO |
| P1 real-shape smoke | **NOT RUN** | 需要每 adapter 8 examples / 3 steps | 尚无真实 shape、gradient 或 peak-memory 证据 |
| P2 baseline fidelity | **NOT RUN** | 需要 BIT validation behavior 与一个 pinned UMBRAE/BrainHub checkpoint | 尚不能声称新目标建立在已复现 baseline 上 |
| P3 interaction-existence | **NOT RUN** | 需要 held-out matched interaction 对 shuffled-brain/phoneme exact null 与跨 session stability | 尚无真实 neural-recoverability 证据 |
| P4 mechanism pilot | **NOT RUN** | 需要六臂 matched-compute、held-out brain benefit | 尚不能声称 NRA 优于 simpler gates |
| P5 confirmatory | **NOT RUN** | 需要 seeds 41/42/43、`K≥4`、bootstrap 与全部 falsification controls | 尚无可发表的 B2T24/B2T25/UMBRAE 结论 |
| Real pipeline components | **NOT IMPLEMENTED** | 计划需要 dataset runner、BIT/UMBRAE model wiring、7B/其他 decoder、trainer、Hungarian evaluator 与 bootstrap inference | 不能把 manifest/config contract 称为 adapter、model 或 benchmark ready |

## Synthetic JSON 的正确读法

synthetic CLI 必须输出：

- `claim_scope: synthetic_execution_only`
- `benchmark_claim: false`

CLI 的 synthetic execution 仅属于 `brain-train`；`brain-evaluate` 不接受 synthetic 参数。各 OPSD arm 使用当前 brain-only student 的 detached greedy rollout；legacy gate 加权的是与 vanilla 相同、未经重建的 privileged-teacher KL。

NRA smoke 采用刻意 favorable 的 controlled null fixture：同一个 frozen teacher、99 个 pronunciation-label donor、3 个 held-out calibration neural record；current/calibration 都是 train，但 fold/session/trial/storage 隔离，各 cohort 内 subject/session/`reliability_stratum` 一致。给定 seed 排列全部 99 个 cyclic assignments，它们不是 99 个独立随机 permutations；exact signed-Fisher statistic 登记 `statistic_epsilon=1e-8`，阈值取 `0.95` higher quantile，98 个 assignment 破坏 anchor pairing，3 个任一格含 anchor label，1 个保持 anchor match。

允许记录的结论只有“指定代码路径在给定 seed/steps 下执行，并满足被测试的不变量”。这个 null fixture 不能称为真实 cross-fit benchmark calibration 或 neural evidence。`losses`、`coverage`、`target delta` 等字段没有真实单位，也没有真实数据 sampling distribution；它们不得改名成 WER/IoU、写入 benchmark matrix，或用于跨方法排名。

## 真实结果最小登记项

当未来真实运行发生时，每条状态更新至少要附：

1. dataset release/split、subject/session 范围和数据许可；
2. dataset 目录、checkpoint 文件、固定 upstream revision、实际 SHA-256 与产生 checkpoint；
3. git revision、resolved config、hardware profile、seed；
4. official-split pool、split/fold/subject/session/`reliability_stratum` matching、prefix/alignment/vocabulary support、negative sampler、`K` 与 duration tolerance；
5. calibration/reliability/stability provenance 与 null permutation count；
6. official task metric、correct-vs-wrong brain gap、相对 CE difference-in-gap；
7. zero/shuffled brain、shuffled privilege、random reward 和 matched-compute baselines；
8. peak allocation、shortfall/coverage、invalid/skip rate；
9. 对应 gate 的 GO/STOP 判断与 bootstrap 方案。

在这些信息齐全以前，状态仍应写作 **NOT RUN** 或 **incomplete**，而不是“趋势良好”。
