# Stage 24：Stage 23 的 encoder padding 评分修复

## 问题与范围

旧 `_single_predictions` 动态补齐输入，但没有向 T5 传入 encoder
`attention_mask`。同一样本与不同长度的邻居组合时可能得到不同预测。
这不是“batch size 不同一定有错”，而是缺少 padding 屏蔽导致 batch 依赖。

本阶段单独输出，不修改旧评分函数、Stage 23 源代码、预注册或历史产物。
只重评 Stage 23 的六个 Development 上下文：两个数据集 × seeds 41/42/43，
原来三个请求、局部/匹配/对照样本全部不变；Amazon λ=48、ML1M λ=64，保留 λ=0。
不训练、不重新构建 impact bank、不调参、不读 FinalTest，不自动修改论文。

缺陷不能解释参数空间的近似误差；修复后的行为结果也不是隐私删除保证。
本轮不覆盖历史基线、排名、攻击评估；这些协议应另行审计，不能宣称已全部修复。
如果旧 Development 校准评分也受影响，冻结的 λ 在新评分协议下是否仍最优尚未验证；
本轮仍固定 λ，以隔离评估实现变化，不将新评分悄悄用作调参依据。

## 固定检查

1. 验证 Stage 23 完整产物、源文件、历史权重和配置哈希。
2. 在 CPU 上重组保存的单用户/双用户/OracleSum/BOTA 端点，逐个核对旧端点张量哈希。
3. 对每个端点，在输入长度和用户身份选出的至多 32 个样本上检查：
   batch size 1/4/8 × 长度排序/短长交错两种分组。覆盖不同 padding 长度、邻居及位置。
4. 每个端点同时运行“不加 mask”负对照。负对照没有观察到差异并不构成失败，
   因为不是每个模型/样本都必须表现出明显 padding 敏感性。
5. 修复版概率容差 `atol=1e-5, rtol=1e-5`；answer loss 容差 `atol=1e-4, rtol=1e-5`。
   允许浮点舍入误差，不要求不同 batch 的 GPU 计算逐位一致。
6. 全队列修复评分后，再将探针与其 batch=1 参考核对，检查真实队列邻居变化。
   任一 gate 失败都停止，不输出该端点的正式预测；保留诊断，不能事后随意放宽容差。
7. yes/no token、teacher forcing、softmax、答案损失和指标公式保持原样，
   唯一评分修正为 encoder attention mask。不引入 decoder mask 或新评价目标。

## 执行

在原 pytorch 环境、`C:\python\BOTA` 目录运行：

```powershell
.\scripts\bota_if\run_ali_stage24_masked_rescore_v1.ps1 `
  -Mode Full `
  -Dataset All `
  -Seed All `
  -Workspace C:\python\E2URec-main
```

可先用 `-Mode Preflight`，只检查文件不加载模型；`-Mode Freeze` 冻结协议而不评分。
可通过 `-Dataset Amazon -Seed 42` 先跑一个上下文；之后用 All 复用该结果。
模型需要 GPU，但 optimizer step 为零。正式 T5 GPU 一致性需要实际运行确认。

Ctrl+C 后重复同一命令：已完成且哈希验证通过的端点/审计跳过，未完成单元重启。
当前端点评分中断时只重算该端点。`.work` 的诊断保留，不自动删除。
不同时启动两个进程；执行锁会阻止它们写同一输出目录。
若是已记录的 audit FAIL，重复命令不会绕过失败，需要先诊断。

## 输出

`C:\python\E2URec-main\outputs\ali_stage24_masked_rescore_v1\summary`

- `REPORT.md`：每个请求/方法的旧、新 E_U 与 residual，完整保留所有种子。
- `behavior_metrics.csv`：修复后的全部行为指标和分组。
- `legacy_vs_masked_metrics.csv`：逐行配对的旧值、新值和差值。
- `batch_invariance.csv`：修复检查与旧评分负对照。
- `prediction_changes.csv`：相同模型和样本的预测/答案损失变化。
- `parameter_decomposition_unchanged.csv`：明确标注为沿用的参数诊断。

报告包含绝对误差和 D_U，避免只看小分母放大的 residual。
当前 audit 容差是工程一致性门槛，不是小删除效应的统计精度保证；分析时还应对照
实际最大 batch 误差与各请求 D_U/E_U 的量级，再决定可解释的精度。
