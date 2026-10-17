# Stage22：IFRU 数值诊断（第一步，不是新增性能比较）

## 冻结范围与边界

这是看到 Stage21 两个求解失败后设计的**事后原因诊断**，不是独立验证，也不声称在历史结果出现前预注册。
新增配置在 Stage22 的模型运行前冻结。历史代码、预注册、失败结果、论文数据均不修改。

- Amazon seed43，原 A1/A2 两个失败请求 + A3 一个成功对照，不换用户。
- 原 Original / fixed-A 坐标；原 1024-sample GGN panel、panel seed43、batch size8。
- 原 12 次上限 power iteration、0.1 相对阻尼；不以诊断结果选择新阻尼。
- 原相对容差 1e-4、绝对容差 1e-10、总 CG 迭代上限 40（重启不重置预算）。
- 仅训练数据上的梯度/曲率诊断；不构造 Development 或 FinalTest 评分加载器。
- 不训练、不改变任何参数，不生成/应用 adapter；保存的解向量只供数值分析。
- 所有成功、失败、上限停止均保留；不把修复分支替换进原基线表。

## 三个固定分支

1. `legacy`：复现旧 CG 的递推、算子调用序列和保护阈值，但失败时保留向量及完整残差记录。
2. `monitor`：每次迭代额外算真实残差 `b-(G+dI)x`，不改变方向更新或原停止判据。
3. `restart5`：每 5 次迭代或发现虚假递推收敛时，以真实残差替换递推残差并重启方向；只有真实残差过门槛才宣布收敛。

`legacy` 仍记录旧实现允许的未收敛截断解，但明确区分 historical acceptance 与 true tolerance satisfied。
监测分支若改变 legacy 结果，需先检查算子可重复性，不能直接做因果解释。

另做原始 GGN 的重复性、加性、对称性、方向 Rayleigh 值，以及 1e-4/1e-2/100 缩放的齐次性探针。
记录真实请求梯度尺度下的齐次误差、参数 dtype、CG dtype、panel 梯度范数和 loss。
这些是有限方向数值探针，不是 PSD/线性性证明；采样梯度范数不能直接证明整个目标未收敛。
本轮**不运行完整 FP64 T5**。FP64 CG/累加不等于 FP64 HVP；缩放探针用于发现尺度敏感性，不能单独证明浮点误差是唯一原因。

## 运行

在 `(pytorch) PS C:\python\BOTA>`：

```powershell
.\scripts\bota_if\run_ali_stage22_ifru_numerical_v1.ps1 -Mode Preflight -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage22_ifru_numerical_v1.ps1 -Mode Smoke -Workspace C:\python\E2URec-main
.\scripts\bota_if\run_ali_stage22_ifru_numerical_v1.ps1 -Mode Full -Request All -Workspace C:\python\E2URec-main
```

Full 自动冻结配置/源代码/输入/运行环境哈希后执行，不必另跑 Freeze。
Preflight 只校验，Smoke 仅 CPU 合成测试；Full 才加载真实模型并使用 GPU。
可先 `-Request A3` 跑成功对照，再用 `-Request All` 自动补齐。不要同时启动两份。

停止后重复原命令：已完成 setup、solver variant、probes 按哈希复用；**正在执行的单元从头重跑，不是逐迭代续算**。
失败数值结果也是有效诊断产物，会复用而不会被反复计算到“成功”。中断/环境错误的 `.work` 保留供检查。
禁止继续修改冻结的脚本/输入后强行复用；真正的实现修复应另行记录或新版本。
本阶段不会删除旧输出。短 `.work` 路径避免之前的 Windows 路径长度问题。

## 产物

`C:\python\E2URec-main\outputs\ali_stage22_ifru_numerical_v1`

- `protocol/result.json`：冻结配置、输入来源、源代码及环境。
- `requests/A*/setup/`：固定请求、模型坐标、panel、梯度和阻尼。
- `requests/A*/legacy|monitor|restart5/result.json`：每步残差、真实阈值比、pAp、重启、失败原因、时长。
- `requests/A*/probes/result.json`：算子探针。
- `summary/solver_diagnostics.csv`、`summary/REPORT.md`、`summary/result.json`：全部九个求解单元。

检查：

```powershell
.\scripts\bota_if\run_ali_stage22_ifru_numerical_v1.ps1 -Mode Analyze -Workspace C:\python\E2URec-main
```

## 如何解释

- `true_over_threshold=1.02` 仅高于阈值 2%，不能形容为数值崩溃；先看量级。
- legacy 失败、restart5 达标：支持此请求存在可改善的数值递推问题，**不证明遗忘质量提高**。
- 旧成功对照不再复现：先查环境、精度、算子，再讨论跨种子机制。
- restart5 仍不达标：本轮到此停止，不放宽容差、不继续扩大迭代预算、不扫新阻尼。
- endpoint GGN inverse 与有限 200-step AdamW replay 的目标并不等价；本轮只记录该区别，不定量归因目标差距。
- 不由这三个请求外推所有数据集，不由 IFRU 推断 BOTA 的失败原因，更不推出隐私保证。

## 开销

不做训练或完整推荐评估。原单次 GGN 调用约 25 秒。
legacy 最多 41 次，monitor/restart5 各最多 81 次；三请求加 power iteration、探针约 669 次上限量级，即约 4.7 小时纯算子时间（不是承诺）。
通常提前收敛会明显缩短。GPU负载/环境影响时间；另有加载、hash 和 panel 梯度开销。
第一项完成后，再判断是否值得实现单用户/双用户分解与删除幅度曲线。
