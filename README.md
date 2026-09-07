# Brain Evidence：脑证据约束的 OPSD / RLVR 研究骨架

本仓库用于检验一个严格的问题：脑条件生成方法的改进，是否真的来自正确的脑试次，而不是语言先验、特权标签复制，或奖励投机。当前交付是 **research skeleton + executable contracts**：包含纯张量算法、contract/config 校验、manifest 准备和 CPU synthetic smoke；它不是可在真实数据上端到端训练或评估的 pipeline。核心设计方法是 Neural-Recoverability-Aware OPSD（NRA-OPSD），同时登记 CE、vanilla OPSD、legacy dual-cosine、RLVR 等计划对照。

> **结果边界**：仓库内置的 synthetic 输出只验证 CPU 代码路径、有限损失、梯度隔离与机制不变量，是 **smoke test，不是 B2T24、B2T25 或 BrainHub/UMBRAE 的实验结果**。三套真实数据实验当前均为 **NOT RUN**；仓库不提供私有数据、外部权重，也不声称任何 SOTA 数字。

详细说明：

- [算法与四格设计](docs/algorithm.md)
- [基准矩阵、基线、指标和停止门](docs/benchmark_matrix.md)
- [完整复现命令](docs/reproduction.md)
- [实验状态与可声明范围](docs/experiment_status.md)

## CPU 快速开始

需要 Python 3.10+。先从 PyTorch 的 CPU wheel 索引安装 CPU 版，再安装本项目；这可避免在仅做本地验证时意外解析 CUDA wheel。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

运行单元测试、全部测试和静态检查：

```bash
python -m pytest tests/unit -q
python -m pytest -q
python -m ruff check .
```

## 合成冒烟验证

整套 CPU smoke：

```bash
python scripts/smoke_test.py
```

单独检查一个确定性 synthetic 方法：

```bash
brain-train --synthetic --method nra_opsd --seed 41 --steps 3
```

CLI 的 synthetic 入口只有 `brain-train --synthetic`；`brain-evaluate` 不运行 synthetic 或真实评估，只接受下文的 config dry-run。synthetic 中各 OPSD arm 都先由**当前** brain-only student 做 greedy rollout，并立即 detach 该 prefix；teacher 随后在同一 prefix 上计算。`legacy_dual_cosine` 只生成 scalar gate，仍对 vanilla arm 那个未经重建的 privileged-teacher KL 加权。

NRA smoke 的 null 是刻意有利、完全受控的实现夹具：同一个 frozen synthetic teacher 处理 99 个 pronunciation-label donor 与 3 个 held-out calibration neural record。current 与 calibration 都标为 train，但 fold、session、trial 和 tensor storage 隔离；每个 cohort 内 subject、session 与 reliability stratum 一致。给定 seed 只打乱 99 个 cyclic shift 的执行顺序；这些是覆盖全部 shift 的 99 个 cyclic assignments，**不是 99 个独立随机 permutations**。阈值是 exact `signed_fisher_alignment`（登记 `statistic_epsilon=1e-8`）的 `0.95` higher quantile；其中 98 个 assignment 破坏 anchor pairing，3 个任一格含 anchor label，只有 1 个保持 anchor match。这个 favorable fixture 只验证统计量与 provenance wiring，不是真实 cross-fit neural calibration、benchmark result 或神经证据。

输出中的 `claim_scope=synthetic_execution_only` 与 `benchmark_claim=false` 是结果解释的一部分。loss、coverage 或 target delta 只用于检查实现是否执行以及机制不变量是否成立，不得放入真实基准表。

## 配置 dry-run

以下命令只解析、合并并校验配置；`adapter.manifest` 仍为 `null`，不会读取私有数据或加载外部 checkpoint。

```bash
brain-train --config configs/b2t24/nra_opsd.yaml --dry-run
brain-train --config configs/b2t25/nra_opsd.yaml --dry-run
brain-train --config configs/brainhub/umbrae_grounding.yaml --dry-run
```

`brain-evaluate` 当前也只有同一种 config dry-run 能力：

```bash
brain-evaluate --config configs/b2t24/nra_opsd.yaml --dry-run
brain-evaluate --config configs/b2t25/nra_opsd.yaml --dry-run
brain-evaluate --config configs/brainhub/umbrae_grounding.yaml --dry-run
```

resolver 会校验 benchmark、task、adapter 与 method family 是否兼容；visual-grounding RLVR 配置还会校验 dense IoU reward 的 threshold/bonus/invalid-penalty schema 和数值范围。`SUPPORTED_METHODS` 只是允许配置登记的名称集合，不表示相应 trainer 已实现。其中 B2T24/B2T25 使用 16 GiB profile 和 14.5 GiB 峰值分配门，UMBRAE 使用 24 GiB profile 和 22 GiB 门。通过 dry-run 只代表配置可解析，不代表模型可复现、trainer 可用或实验已完成。

## 准备用户提供的真实数据

准备脚本只把明确给出的本地路径写成 adapter 可验证的 manifest，不下载数据或权重。dataset 路径必须是目录；checkpoint 路径必须是单个文件。将示例路径、upstream revision 和 digest 替换为你实际固定的值：

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

脚本会重新计算 checkpoint 文件的 SHA-256，digest 不匹配即失败。`--output` 不得与 checkpoint 文件是同一路径、symlink alias 或 hardlink alias；仓库忽略整个 `/local/`，避免本机路径被提交。manifest 创建不等于基线复现。进入 NRA-OPSD 之前，必须先完成 BIT 验证行为和一个固定 UMBRAE/BrainHub checkpoint 的 fidelity gate。

## 机制 pilot 与 confirmatory 预检

先检查六臂、匹配计算量的机制消融配置：

```bash
brain-train --config configs/experiments/ablation.yaml --dry-run
```

该 YAML 只登记计划中的六臂比较并接受 config 校验；仓库没有这些真实 trainer。配置的 `K=2` 仅用于低成本机制 pilot 设计，它不是 confirmatory 设置。未来真实机制 pilot 必须在留出数据上比较 `uniform`、`reliability`、`teacher_contrast`、`brain_contrast`、`legacy_dual_cosine` 与 `interaction_projection`，并应用 [benchmark matrix](docs/benchmark_matrix.md) 中的 P3/P4 停止门。

三套 confirmatory 配置的预检命令就是上面的三个数据集 dry-run。它们固定 seeds `41/42/43`、`K=4`、至少 `999` 次 exact-statistic null permutation、`alpha=0.05`、anchor-local controls，以及 frozen cross-fitted CE brain-only support。真实 confirmatory 运行必须等 P0–P4 全部通过后再启动；B2T24、B2T25 与 BrainHub/UMBRAE 当前都仍是 **NOT RUN**。

## 方法边界

- student 在训练 rollout 与推理时只接收 brain soft tokens，不接收 phoneme/transcript privilege。
- 现有纯 tensor target/objective 会 detach teacher、frozen CE support、交互向量、可靠性、稳定性、null statistic 和构造 target；在该函数边界内梯度只更新 live student。这不是对尚未实现的真实数据/训练 orchestration 的端到端防泄漏证明。
- real loader 必须先按官方 split 建池；matched-negative trial、负发音来源 trial 和四格 control trial 必须与 anchor 同 split、fold、subject、session 与 `reliability_stratum`；四格还要求 control 的真发音区别于 anchor/negative。
- reliability calibration 必须精确匹配当前 pronunciation ID；validity mask 只能是 boolean 或有限二值 `{0,1}` tensor。
- primary/confirmatory 缺少合格 cross-session support 时 stability 为 `0`、不转移并记录；把缺失当作 `1.0` 只允许在显式 ablation 中。
- NRA 主路径必须同时传入 `NRAEvidence` 与 `NRAInteractionBatch`；两者的 subject/pronunciation/reliability stratum、fold/session/trial、checkpoint、prefix/alignment/support、sampler/aggregation/weight policy、seed 与 anchor-local control policy 必须精确一致。
- `NRAEvidence.statistic_epsilon` 是 exact signed-Fisher calibration identity 的一部分；调用 `recoverable_target` 时的 `epsilon` 必须与它完全相同，否则 fail closed。
- NRA mode 还要求 caller 显式传入 reliability 与 session stability，并用整体 scalar clipping 保持 correction 在 brain-supported span 内；这两个 gate 值进入 projection 时仍是上游已验证的裸 detached scalar/tensor，并没有被绑定成端到端 typed gate artifact。
- NRA 主路径拒绝裸 interaction tensor、单独的 interaction weights 和任何 `null_interaction`（包括零向量）；raw tensor、单独传入的 weights 与 raw-null subtraction 只属于显式的 `interaction_only_ablation`。
- 接受的 student log-probabilities 会先用与 source dtype 和词表大小相适配的 `logsumexp` 容差验证，再在 Fisher 运算前显式归一化；这覆盖 large-vocabulary BF16 输入。`target_kl` 诊断将纯数值舍入导致的负值截到零。
- diagnostics 中 `geometric_coverage` 表示 interaction/projection/`kappa` 的几何支持；`transfer_coverage` 还要求 reliability 与 stability 都大于零，旧字段 `coverage` 是后者的兼容 alias。
- 每个主结果必须同时报告任务指标与正确脑对 matched-wrong brain 的差值，并相对 CE 报告 difference-in-gap。
- 对比必须保持 vocabulary support、forward/token budget、negative sampler、decoding、seeds 与可训练参数预算一致。

真实 dataset runner、BIT/UMBRAE model wiring、7B/其他 decoder、trainer、multi-instance Hungarian evaluator 与 bootstrap inference 尚未实现。它们是 [planned comparison matrix](docs/benchmark_matrix.md) 的研究要求，不是本仓库当前可执行能力。
