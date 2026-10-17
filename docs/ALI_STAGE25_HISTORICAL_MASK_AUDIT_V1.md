# Stage 25：历史主实验与 Development 校准评分审计/修复

## 已定位的缺陷

缺少的是 encoder `attention_mask`，不是笼统的“batch size 不一致”。
动态 padding 未屏蔽时，同一样本会随邻居样本长度而改变预测。

| 历史路径 | 审计结论 | 本轮处理 |
|---|---|---|
| evaluation._single_predictions / _ensemble_predictions | 没传 encoder mask | 独立新评分器；单模型/混合模型均修复 |
| Stage 7–9、12–14 的 Development 校准 | 复用上述评分器，受影响 | ML 冻结主模型、λ=64/0/历史1 控制在原 Dev-Select 重评；不扫描整个网格 |
| Stage 15 Development NDCG | score_model 明确采用 legacy no-mask | 原候选集、相关性、K、tie-break 不变，重评 |
| Stage 16 FinalTest 二分类和排名 | 两条 lane 都经 Stage10/common evaluator | 独立显式确认后按原队列重评 |
| Stage 18 Amazon λ 校准 | common evaluator，受影响 | 固定 λ=48、0，原 K2/K4/full Development 和三种子重评 |
| Stage 20/21 Amazon 比较 | 单模型及集成评分均受影响 | 原 A1–A3、seeds 42/43；Stage20 冻结基线在原 K2/K4/Dev-Select 也重评 |
| Stage 17 行为、排名、MIA 压力测试 | 同一受影响入口 | 已识别，但不在本轮重跑范围，不能沿用为“已修复”的结论 |
| 轨迹 replay / source transport 的 masked_forward | 已传 mask | 不重训、不改参数空间诊断 |

源文件未被热补丁或覆盖，历史预注册与产物全部保留。

## 冻结边界

- ML1M λ=64，Amazon λ=48；保留 λ=0 消融，不重新选择。
- 只处理已冻结的主模型、选定/事先指定的诊断基线与校准请求；不扩展候选参数网格。
- 保留历史基线 eligibility：以前不合格的 PCGrad/SISA 不能因这次评分改善而自动升级为合格基线。
- 工程修复不证明历史调参仍然最优；若排名/效用或最优性结论需要变化，应另行决策。
- 不访问新队列、不替换请求；FinalTest 仅为旧 Stage16 队列的纠错性重评，不包装成新 holdout。
- 所有模型 optimizer updates=0。没有 IFRU 求解、bank transport 或 partition retraining 路径。

## 缺失权重是实际限制

预检共 388 个逻辑单元：370 可重评、16 个历史分片权重未保留、2 个历史数值求解失败。
16 个缺权重单元来自 Amazon SISA/RecEraser：比较 12 个、冻结校准 4 个。
不能从旧预测恢复带 mask 的新预测，因此明确输出 `WEIGHTS_NOT_RETAINED`。
求解失败仍保留 `HISTORICAL_NUMERICAL_FAILURE`，不作为胜出或零误差。

本轮不擅自重训这些模型。若要补齐，需恢复原权重或单独授权按冻结配置重建。
原 hash/manifest 只能验证权重，不可逆推出已清理的张量。

## 执行顺序

在 `(pytorch) PS C:\python\BOTA>`：

```powershell
# 1. Development：244 个单元（226 可评分、18 保留为不可用）
.\scripts\bota_if\run_ali_stage25_historical_mask_audit_v1.ps1 `
  -Mode Full `
  -Scope Development `
  -Workspace C:\python\E2URec-main

# 2. FinalTest：原 Stage16 的 144 个二分类/排名单元
# 独立确认，仅评估已冻结模型，不能依据 Development 修复结果临时换 λ。
.\scripts\bota_if\run_ali_stage25_historical_mask_audit_v1.ps1 `
  -Mode Full `
  -Scope FinalTest `
  -ConfirmFinalTest `
  -Workspace C:\python\E2URec-main
```

`-Mode Preflight` 只做资产/协议检查；`-Mode Freeze` 保存协议和审计报告，不运行 GPU。
可用 `-Lane binary` 先补二分类，再用 `-Lane All` 自动复用已完成单元并补排名。
不要同时启动两条命令；执行锁阻止并发写入。

FinalTest 不加 `-ConfirmFinalTest` 会在数据加载前拒绝执行。
阅读历史队列元数据用于冻结，不等于重新加载 FinalTest 模型评分样本。

## 一致性与断点

每个单模型/集成分片都执行 Stage24 同级别 mask/batch 检查；集成后再核对真实队列邻居。
保留不加 mask 的负对照。原 ensemble 的 token probability mixture、答案损失定义不变；
不能替换为平均已归一化概率或平均分片 loss。Stage15 排名的 float64 归一化约定也保留。

Ctrl+C 后重复同一命令即可：已完成单元校验后跳过，未完成端点重新评分。
即使最后一个单元 audit 失败，Analyze 也不能把它包装成完成。
文件改动、权重改动、runtime 改动将停止执行；不要手动删除协议后继续拼接结果。

## 输出和解释

`C:\python\E2URec-main\outputs\ali_stage25_historical_mask_audit_v1`

- `audit/REPORT.md`：评分路径审计、范围及未修复项。
- `audit/endpoint_inventory.csv`：冻结端点和不可用原因。
- `summary/Development/REPORT.md`、`metrics.csv`、`prediction_changes.csv`。
- `summary/FinalTest/REPORT.md`、`metrics.csv`、`prediction_changes.csv`。
- 每个 `cells/<id>/result.json` 内含 batch/负对照/真实队列 gate、预测与输入身份。

保留 (E_U\)、(D_U\)、E_U/D_U、原论文 local_exact_masked_residual、P5 sensitivity、
AUC/logloss、NDCG/Recall 及能配对的旧值。若原记录没有对应评分，旧值为 NA，而不是伪造。
Development 有缺失基线，因此 `COMPLETED_WITH_UNAVAILABLE` 表示已完成可执行部分，不是完整比较已成立。
没有自动改论文或宣布新结论；完成后需核验绝对误差、数值精度和比较完整性。
