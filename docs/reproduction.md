# 复现手册：从 CPU smoke 到真实实验门控

以下命令均从仓库根目录执行。当前仓库是 research skeleton：可复现范围是纯张量/contract tests、config dry-run、manifest 校验和 synthetic smoke，不包含真实 dataset/model runner。本文把“可本地复现的代码路径”和“需要外部数据/权重及尚未实现组件的真实研究”分开；前者不能替代后者。

## 1. CPU 环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

确认公开命令目标可加载：

```bash
brain-train --help
brain-evaluate --help
python scripts/prepare_b2t.py --help
python scripts/prepare_brainhub.py --help
```

若尚未安装 console script，可用等价模块入口诊断：

```bash
python -m brain_evidence.cli.train --help
python -m brain_evidence.cli.evaluate --help
```

## 2. 测试与 lint

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest tests/test_documentation.py -q
python -m pytest -q
python -m ruff check .
```

这些测试不需要 B2T 或 BrainHub 私有数据。`tests/test_documentation.py` 通过实际调用 CLI/script 与公开 config resolver 验证命令和配置契约；它不 grep 文档措辞。

## 3. Synthetic smoke（仅代码验证）

```bash
python scripts/smoke_test.py
```

也可逐个运行：

```bash
brain-train --synthetic --method ce --seed 41 --steps 3
brain-train --synthetic --method vanilla_opsd --seed 41 --steps 3
brain-train --synthetic --method legacy_dual_cosine --seed 41 --steps 3
brain-train --synthetic --method nra_opsd --seed 41 --steps 3
brain-train --synthetic --method rlvr --seed 41 --steps 3
```

synthetic CLI 只存在于 `brain-train`；`brain-evaluate` 不接受 `--synthetic`。CE smoke 使用 gold prefix；各 OPSD smoke 使用当前 brain-only student 生成并 detach 的 greedy rollout，teacher 和 student 随后共享该 prefix。vanilla arm distill privileged-teacher distribution；legacy arm 只计算 detached scalar gate，再用它加权同一个、未经重建的 privileged-teacher KL。

NRA 的 synthetic null 是刻意有利的 controlled smoke fixture，不是从真实神经数据估计的 calibration：

- current 与 calibration cohort 都标记为 train，并使用同一个 frozen synthetic teacher；两者的 fold、session、trial ID 与 neural tensor storage 不相交；
- 每个 cohort 内的 3 个 neural record（1 anchor + 2 controls）共享 subject、session 和 `reliability_stratum`；calibration 另有 99 个 pronunciation-label donor；
- 给定 seed 对全部 99 个 cyclic shift 排序，因此是 99 个 cyclic assignments，而不是 99 个独立随机 permutations；
- 每个 assignment 用登记的 `statistic_epsilon=1e-8` 计算 exact `signed_fisher_alignment`，阈值取 `0.95` higher quantile；98 个 assignment 破坏 anchor pairing，3 个任一被评分格含 anchor label，1 个保持 anchor match。

验收只包括：loss 有限、结果在相同 seed 下确定、teacher 冻结、梯度只到 student，以及 copier-only/unsupported target 回到 student、supported target 才改变。上述 favorable null 只压力测试 statistic/provenance wiring，不能叫作真实 cross-fit、神经证据或 benchmark calibration。任何 JSON 数值都属于 **synthetic smoke diagnostics**，不是 WER、IoU、leaderboard 或模型质量结果，不能支持 SOTA 或跨方法优劣声明。

## 4. 三套配置的无数据预检

```bash
brain-train --config configs/b2t24/nra_opsd.yaml --dry-run
brain-train --config configs/b2t25/nra_opsd.yaml --dry-run
brain-train --config configs/brainhub/umbrae_grounding.yaml --dry-run
```

预期行为是输出 resolved config，并明确 `status=resolved_only`、`private_data_accessed=false`。校验重点：

- B2T24/B2T25：BIT adapter、phoneme privilege、seeds `41/42/43`、`K=4`、999 null permutations、14.5 GiB allocation gate；
- UMBRAE：BrainHub manifest contract、计划 UMBRAE backend、evidence IoU RLVR 的 dense reward schema/range、相同 confirmatory controls、22 GiB allocation gate；
- 三者：`adapter.manifest=null`，因此 dry-run 不能被解释成真实 adapter 已就绪。

resolver 还会 fail closed 校验 benchmark/task/adapter/method family 组合。例如 Brain-to-Text 配置不能登记 visual-grounding method，反之亦然。`SUPPORTED_METHODS` 只表示 config schema 接受的计划方法名，不表示存在对应 trainer。

如需 evaluator 的同一路径预检：

```bash
brain-evaluate --config configs/b2t24/nra_opsd.yaml --dry-run
brain-evaluate --config configs/b2t25/nra_opsd.yaml --dry-run
brain-evaluate --config configs/brainhub/umbrae_grounding.yaml --dry-run
```

这是 `brain-evaluate` 当前唯一执行模式；它没有 synthetic 路径，也没有真实 evaluator runner。

## 5. 生成本地 manifest

脚本要求 dataset 是已经存在的本地目录、checkpoint 是已经存在的本地文件，并拒绝 URL；它们不会自动下载私有资产。`--revision` 必须由运行者提供真实、固定的 upstream release/commit，下面用必填环境变量而不猜测 revision。`--checkpoint-sha256` 必须是该文件的 64 位十六进制 SHA-256，脚本会重新计算并核对。

```bash
mkdir -p local/manifests

B2T24_CHECKPOINT=/absolute/path/to/bit-b2t24-checkpoint.pt
B2T24_SHA256="$(sha256sum "$B2T24_CHECKPOINT" | awk '{print $1}')"
: "${B2T24_REVISION:?set B2T24_REVISION to an exact upstream release or commit}"
python scripts/prepare_b2t.py \
  --dataset-path /absolute/path/to/b2t24-dataset \
  --checkpoint-path "$B2T24_CHECKPOINT" \
  --revision "$B2T24_REVISION" \
  --checkpoint-sha256 "$B2T24_SHA256" \
  --backend b2t24 \
  --output local/manifests/b2t24.yaml

B2T25_CHECKPOINT=/absolute/path/to/bit-b2t25-checkpoint.pt
B2T25_SHA256="$(sha256sum "$B2T25_CHECKPOINT" | awk '{print $1}')"
: "${B2T25_REVISION:?set B2T25_REVISION to an exact upstream release or commit}"
python scripts/prepare_b2t.py \
  --dataset-path /absolute/path/to/b2t25-dataset \
  --checkpoint-path "$B2T25_CHECKPOINT" \
  --revision "$B2T25_REVISION" \
  --checkpoint-sha256 "$B2T25_SHA256" \
  --backend b2t25 \
  --output local/manifests/b2t25.yaml

UMBRAE_CHECKPOINT=/absolute/path/to/umbrae-checkpoint.pt
UMBRAE_SHA256="$(sha256sum "$UMBRAE_CHECKPOINT" | awk '{print $1}')"
: "${UMBRAE_REVISION:?set UMBRAE_REVISION to an exact upstream release or commit}"
python scripts/prepare_brainhub.py \
  --dataset-path /absolute/path/to/brainhub-dataset \
  --checkpoint-path "$UMBRAE_CHECKPOINT" \
  --revision "$UMBRAE_REVISION" \
  --checkpoint-sha256 "$UMBRAE_SHA256" \
  --backend umbrae \
  --output local/manifests/umbrae.yaml
```

示例输出位置 `local/manifests/` 不是正式 config 的隐式默认值。准备脚本会在写入前拒绝 `--output` 与 checkpoint 的同路径、symlink alias 或 hardlink alias；整个仓库根目录 `/local/` 已被忽略。应复制正式 YAML 为一个不提交的本地配置，将 `adapter.manifest` 显式指向生成文件，并再次执行 dry-run。不要把机器路径或受限资产写回共享 confirmatory YAML。

## 6. Mechanism pilot 预检

六臂消融配置：

```bash
brain-train --config configs/experiments/ablation.yaml --dry-run
```

YAML 登记的计划比较是 `uniform`、`reliability`、`teacher_contrast`、`brain_contrast`、`legacy_dual_cosine`、`interaction_projection`；seeds `41/42/43`；`K=2`；duration tolerance `0.15`；mean aggregation；one rollout；max generation tokens `128`；16 GiB profile，estimated peak 14.0 GiB。解析这些字段不代表六个 trainer 已实现，也不是一次 pilot execution。

这条命令只是配置预检。真实 P3/P4 pilot 还需要：

1. P2 已复现 BIT validation behavior 或 pinned UMBRAE/BrainHub checkpoint；
2. 训练 fold 内生成 exact-statistic calibration artifact，pilot 至少 99 permutations、`α=0.05`；
3. real loader 先按官方 split 建池；matched-negative、`p_j` 来源 trial 与四格 control 都同 split/fold/subject/session/`reliability_stratum`，`p_j` 不同，`p_c` 区别于 anchor/negative，并记录 effective `K`；
4. 在 held-out 数据上同时计算 official task metric 与 correct-vs-matched-wrong brain gap；
5. 应用 [P3/P4 停止门](benchmark_matrix.md#5-顺序执行与-stopgo-gates)。

当前仓库没有把上述真实 pilot 标记为已运行。

## 7. NRA typed artifact 边界

真实 NRA 调用不是“给 projection 一个 tensor 和阈值”。`recoverable_target(mode="nra", ...)` 同时要求：

- `NRAEvidence`：保存 frozen cross-fitted CE 的 brain-only direction、training-only exact signed-Fisher null threshold、生成该 calibration statistic 的 `statistic_epsilon`、producer membership、checkpoint，以及 `K`、sampler、aggregation、weight、prefix/alignment/support、seed 和 control policy；
- `NRAInteractionBatch`：保存当前 interaction tensor 及同一批次的 subject、pronunciation、reliability stratum、fold/session/trial、checkpoint 和相同 statistic context。

构造 target 前，接口逐字段比较二者，并检查实际 negative axis 等于登记的 `K`、aggregation 相同、uniform-weight policy 可由实际权重验证；调用 `recoverable_target` 的 `epsilon` 还必须与 artifact 的 `statistic_epsilon` 完全相同。任何 mismatch 都 fail closed。NRA 模式不接收裸 interaction tensor、独立 `interaction_weights` 或 `null_interaction`；即使 `null_interaction` 是全零也会拒绝。raw-null subtraction 仅可在显式 `mode="interaction_only_ablation"` 中用于低层消融，不能替代 `signed_fisher_alignment` 的 permutation threshold。

reliability/null calibration 还要求记录的 pronunciation ID 与当前 pronunciation 精确相同，运行时 provenance/policy 字符串非空；validity mask 只能是 boolean 或有限二值 `{0,1}` tensor。这些都是 fail-closed contract，不代表真实 calibration artifact 已生成。

NRA mode 不为 reliability 或 session stability 提供隐式默认值；caller 必须显式传入两者。projection 会 detach/校验这些值并用整体 scalar clipping 保持 correction 在 brain-supported span 内，但当前接口接收的仍是上游已验证的裸 scalar/tensor，而不是和 `NRAEvidence` 同级的 typed gate artifact。没有真实 orchestration 时，不能把局部 tensor detachment 写成整个 pipeline 已经 leakage-proof。

读取 projection diagnostics 时，`geometric_coverage` 只表示 interaction/projection/`kappa` 在几何上可支持；`transfer_coverage` 还要求 reliability 和 stability 都大于零。兼容字段 `coverage` 等于 `transfer_coverage`，因此 gate 为零时不会再被读作有效转移。

student log-probabilities 会先在内部工作 dtype 中做 vocabulary-aware `logsumexp` 检查，容差考虑 source dtype 与词表大小；通过后再减去 log-mass，并以归一化概率做 Fisher 运算。该 contract 支持 large-vocabulary BF16，同时拒绝实质性未归一化输入。`target_kl` 诊断将浮点舍入导致的微小负数 clamp 为零；这只是数值报告规则，不是额外训练信号。

## 8. Confirmatory 预检与执行前清单

再次运行三条正式 dry-run：

```bash
brain-train --config configs/b2t24/nra_opsd.yaml --dry-run
brain-train --config configs/b2t25/nra_opsd.yaml --dry-run
brain-train --config configs/brainhub/umbrae_grounding.yaml --dry-run
```

只有满足以下条件才进入 P5：

- P0–P4 依次通过，没有越过 baseline fidelity 或 interaction-existence gate；
- frozen cross-fitted CE support checkpoint、reliability/calibration artifacts 与 stability bank 都固定 provenance；
- primary/confirmatory 缺少合格 cross-session support 时 stability fail closed 为 `0`、不转移并记录；`1.0` 只能属于显式 ablation；
- `K≥4` 且 negative pronunciations distinct；shortfall 样本跳过并记录；
- exact-statistic null 至少 999 permutations、`α=0.05`；
- seeds 固定 `41/42/43`，相同 vocabulary support、forward/token budget、sampler 与 decoding；
- B2T 使用 paired bootstrap，BrainHub 使用 unique-image cluster bootstrap；
- 16 GiB/24 GiB allocation 分别不超过 14.5/22 GiB。

当前实现只保证纯张量算法、contract/config 解析、adapter manifest 验证与 synthetic execution contract。真实 dataset runner、BIT/UMBRAE model wiring、7B/其他 decoder、trainer、multi-instance Hungarian evaluator 与 paired/cluster bootstrap inference 均尚未实现；数据许可和外部 checkpoint 也由运行者负责。B2T24、B2T25 与 BrainHub/UMBRAE 真实实验均为 **NOT RUN**。因此本文不伪造一个会把 dry-run 冒充训练或评估的命令。

## 9. 一次完整的本地提交前验证

```bash
python -m pytest -q
python -m ruff check .
python scripts/smoke_test.py
brain-train --config configs/b2t24/nra_opsd.yaml --dry-run
brain-train --config configs/b2t25/nra_opsd.yaml --dry-run
brain-train --config configs/brainhub/umbrae_grounding.yaml --dry-run
brain-evaluate --config configs/b2t24/nra_opsd.yaml --dry-run
brain-evaluate --config configs/b2t25/nra_opsd.yaml --dry-run
brain-evaluate --config configs/brainhub/umbrae_grounding.yaml --dry-run
git diff --check
```

只有这些命令本次真实返回成功时，才可以声明本地代码与配置 gate 通过；它们仍不产生真实 benchmark result。
