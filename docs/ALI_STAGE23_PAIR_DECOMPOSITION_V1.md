# Stage23：跨数据集／种子的单用户与双用户误差分解

## 目的和研究边界

这是看到已有结果后设计、在本阶段模型运行前冻结的**事后机制诊断**，不是全新独立验证。
先解释 BOTA；本轮不运行 IFRU，不用 BOTA 的分解结论代替 IFRU 的机制证据。

区分三个量：

- 单用户传输近似误差：BOTA 单用户变化与精确单用户变化相差多少。
- 联合删除非加性：联合精确变化与两个单用户精确变化之和相差多少。
- 行为影响：这些参数差异是否对应局部预测误差，还是在预测上几乎无影响。

不调 λ、不筛成功请求、不访问 FinalTest、不覆盖旧论文指标、不做新分组算法、不声称隐私保证。

## 固定范围

| 数据集 | 种子 | 三个 K=2 请求 | BOTA λ | 对照 |
|---|---|---|---:|---|
| Amazon | 41/42/43 | 原 Stage20/21 A1、A2、A3，同一批用户 | 48 | λ0 + Exact 单／双用户 + Original |
| MovieLens-1M | 41/42/43 | 既有 L4M4 中按原 public hash 对应真实用户、分低／中频排序后取前三对 | 64 | 同上 |

ML1M 固定配对：M1=(2070,2898)，M2=(3089,4116)，M3=(2018,3462)。
Amazon 固定配对：A1=(702,758)，A2=(302,354)，A3=(132,64)。
所有配对由元数据决定；不读取新预测结果来选用户。
不通过 `forget_train_indices` 的位置与 public metadata 直接 zip 推断用户身份；使用可验证用户哈希关联身份。

Amazon seed41 在这三个固定 A 请求上新增评估，用于同请求跨种子诊断；它不等于重跑旧 Stage19 的 R01–R12，不能声称已经解释旧 R 请求的极端值。
所有三个种子的全部结果和失败都必须保留，不用“多数种子有效”概括未通过的单元。

## 可复用与新增计算

- 所有六个 dataset–seed context 先重新进行一次 200-step **无删除重放**，必须逐张量等于历史 Original；不一致即停止。
- 所有单用户 Exact 使用同一训练起点、fixed-A、种子、批次顺序、AdamW 设置、200 步、原 slot-preserving loss 分母。
- Amazon seed42/43 的六个联合 Exact 经 manifest、坐标、请求身份和 Original 重放检查后复用。
- Amazon seed41 三个联合 Exact 和 ML1M 九个联合 Exact 新计算。
- ML1M 复用 Stage17 的用户级 FS0/FS64 bank；Amazon 重建原传输的用户级 bank，保存单用户向量，并要求 seeds42/43 的联合 BOTA adapter **逐张量复现** Stage20/21。
- Amazon bank 继续运行原 lambda1 内部实现控制，但它不参与结果比较或选择。

新增：36 次单用户 Exact、12 次联合 Exact、6 次无删除控制、3 次 Amazon bank。
合计 54 次普通 200-step 重放 + 3 次 200-step transport = 11,400 个物理 optimizer step calls。
transport 每步成本高于普通重放，不能按 step 数直接比较总时间。

## 精确误差恒等式

令 C 为同种子的 Original，U/V/UV 为精确单用户／联合删除端点，A_u/A_v/A_uv 为对应近似端点。

```
I_uv = (UV-C) - (U-C) - (V-C)
e_u  = A_u-U
e_v  = A_v-V
r    = A_uv-A_u-A_v+C       # 实际 float32 发布／组合的舍入差
e_uv = e_u + e_v - I_uv + r
```

保存各项绝对范数、相对范数、方向余弦、恒等式闭合误差，以及平方误差分解的所有有符号交叉项。
**不能把各项范数的比值当成“原因贡献百分比”**；单用户误差可能互相抵消，也可能抵消交互项。
参数差异全部在同一个模型、同一 fixed-A 坐标中计算，不跨种子直接比较向量方向。
精确目标范数不足 1e-12 的比值输出 null 并保留绝对量，不用任意放大后的 residual 判断。

## 行为对照

实际构造并评估 `OracleSum = C + (U-C) + (V-C)` 的模型。
它回答“假如单用户变化完全准确，直接相加还会错多少”。
**不是把两个模型的概率相加**；先组合参数再用同一评分函数推理。

每个 context 评估 Original、6 个单用户 Exact、3 个联合 Exact、3 个 OracleSum、两种 λ 下各 6 个单用户和 3 个双用户 BOTA，共最多 31 个端点。
同一 context 所有端点都使用相同 Development 样本、标签、顺序和 batch size4。

- 全部请求用户 Development 样本：Amazon 30，ML1M 236。
- 每位用户固定哈希选 5 条的等样本量辅助视角：不按标签／预测挑选。
- 固定哈希选择 512 条非请求用户样本作为诊断对照。
- 最终推理查询数：Amazon 542／端点，ML1M 748／端点。
- 单用户结果在本人样本上比较本人 Exact；双用户结果在联合样本上比较联合 Exact，同时报告逐用户结果。
- E_U、D_U、relative residual 和 Toward-Exact Rate 全部保留，避免只看归一化结果。
- 512 条诊断对照**不等于正式 aggregate utility/NDCG/MIA 评估**，不能用于更新主论文效用结论。

## 如何运行

在 `(pytorch) PS C:\python\BOTA>`：

```powershell
.\scripts\bota_if\run_ali_stage23_pair_decomposition_v1.ps1 `
  -Mode Full `
  -Dataset All `
  -Seed All `
  -Workspace C:\python\E2URec-main
```

Full 自动 Freeze。也可先只跑 Amazon seed43（与 Stage22 直接相接）：

```powershell
.\scripts\bota_if\run_ali_stage23_pair_decomposition_v1.ps1 -Mode Full -Dataset Amazon -Seed 43 -Workspace C:\python\E2URec-main
```

后续用 All 补齐，不会重新计算已完成单元。只完成部分时返回 INCOMPLETE，表示仍有计划单元未跑，不是该次执行失败。

```powershell
.\scripts\bota_if\run_ali_stage23_pair_decomposition_v1.ps1 -Mode Preflight -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage23_pair_decomposition_v1.ps1 -Mode Smoke -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage23_pair_decomposition_v1.ps1 -Mode Analyze -Workspace C:\python\E2URec-main
```

Preflight 只检查来源和元数据；Smoke 为 CPU 合成测试，不加载真实模型。
不要同时启动两份。停止后重跑同一命令：完成单元按哈希复用；**中断的重放／bank／预测单元从头执行，不是逐步续算**。
不自动删除 `.work`；错误记录保留。训练参考失配、请求身份错误、OOM 属执行错误，停止检查，不当成方法负面结果。
lambda48/64 transport 数值失败是正式诊断结果，保留失败，继续能完成的参考与 λ0，不缩小 λ 来“救结果”。

## 输出

`C:\python\E2URec-main\outputs\ali_stage23_pair_decomposition_v1`

- `protocol/result.json`：冻结计划、全部上下文、源代码／输入哈希与环境。
- `Amazon_s41/` 等六个目录：control、bank、exact、parameters、predictions、behavior。
- `summary/parameter_decomposition.csv`：全部 36 个请求×种子×λ 参数分解单元（含失败）。
- `summary/behavior_metrics.csv`：单用户／联合／OracleSum 的分组预测误差（含不可用行）。
- `summary/result.json`：完整交叉项与所有细节；`summary/REPORT.md` 为索引。

历史文件与论文文件均不修改。

## 解释顺序与限制

1. 先看无删除与历史 BOTA 重现控制是否通过。
2. 看单用户误差，随后看 I_uv 及误差方向，不只比较两个范数大小。
3. 看 OracleSum 是否接近联合 Exact：验证非加性是否影响实际预测。
4. 对照 λ0/冻结 λ，判断校准改善的是单用户误差还是通过误差抵消改善组合。
5. 按同一请求跨种子报告，再按数据集描述分布；不要把 18 个相关单元当成 18 个独立数据集做显著性宣称。

跨数据集仍有混杂：Amazon 每用户 window exposure=2、ML1M=1；原始模型、训练数据量、优化器配置和校准 λ 也不同。
跨种子同时改变 fixed-A 和训练 RNG，不能将差异单独归因于“随机数种子坏了”。
本轮是诊断线索，不证明非线性是唯一原因，也不直接得到可部署的风险检测器、分组规则或隐私保证。
删除幅度曲线仍属于后续独立阶段，本轮不自动执行。

## 时间估算

历史同机普通 200-step replay 约 29 秒，Amazon 六用户 bank 约 325 秒。
据此训练／transport 主体约 42 分钟；再加 186 个端点的受限 Development 推理、模型加载、I/O 和哈希。
建议先预留约 1–2 小时；这是历史速度外推，不是实测承诺。若有 GPU 竞争或加载缓慢会延长。
