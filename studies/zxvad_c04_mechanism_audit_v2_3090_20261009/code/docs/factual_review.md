# zxVAD规划文件事实与机制审核

审核日期：2026-10-09（北京时间）。对象为用户提供的 `D:/Browser/zxVAD_critical_review_and_research_plan_for_Codex.md`；文档中的执行建议仅作待审核草案。本审核没有运行GPU训练或目标评测。论文依据为已缓存的MERL作者版本、CVF补充材料；代码依据为已审核c04双卡筛查的冻结源码。

## 裁决

| 项目 | 裁决 | 应实施的范围 |
|---|---|---|
| P0-1 NC类别与生成来源混杂 | **GO** | 先进行源域冻结模型2×2诊断，再做来源匹配训练消融；不能预设该混杂已造成跨域损失。 |
| P0-2 训练监督与最终评分关系 | **MODIFY / GO** | 研究NC引导是否形成PSNR可利用的区分、NC读出是否互补；删去任何“NC对G没有梯度”的表述。 |
| CNG-FP异常干预→干净未来 | **MODIFY / GO** | 可检验它在固定c04跨域任务中的增益与退化，须有等预算常规增强、静态干扰、时序随机控制；不能把基础目标和移动patch称为新方法。 |
| 没有源域机制结果前叠加复杂结构 | **STOP** | 当前没有必要进行Transformer/多memory/外部语义模型堆叠或最优组合搜索。 |
| 增加AUPRC、事件级指标、报告bootstrap | **不采纳为性能依据** | 用户已固定AUROC单指标。源域误差、梯度、注意力及运行时间只能列为机制诊断。 |
| STG实现、目标专用权重调参 | **不在此分支执行** | STG由主分支处理；其β和监督条件应透明列注，不能简单用协议差异抹去强基线。 |

## 需要保持或修正的事实

1. **NC的正负样本来源确实不同。** zxVAD的式(2)–(5)使用生成器预测正常帧与将供体片段粘贴到真实输入正常帧形成的伪异常。文档把它视为可检验捷径风险是合理的；不能进一步认定分类器一定依赖模糊或边界。当前c04具体使用`pred.detach()`作为NC正样本，`paste(clip[:,0], donor, ...)`作为负样本，因此来源和时间位置都不同。

2. **NC通过可微引导作用于G。** 论文式(6)的G损失含`α_N E[0.5(N(G(X))−1)^2]`；NC自己的四项损失并非直接加到G上。当前`baseline_model.one_update`在G更新时冻结N的参数，但没有detach预测图；其audit断言引导对G的梯度非零。随后训练N才使用`pred.detach()`，使NC参数更新不回传G。这是正确的两阶段图隔离，而非监督链条断开。P0-2应称“引导与PSNR读出的作用关系未被证实”。

3. **参数冻结与BN冻结不同。** c04在G引导时保留N的train模式；每step共有4次N前向更新BN计数。改变成N.eval会改变该基线训练过程，需作为独立消融。目标测试的N读出必须eval且不更新任何权重或运行统计。

4. **Attention定义仍含独立实现选择。** 论文说明从N最后卷积的feature maps做SCDA，正常attention监督为全1，伪异常为粘贴mask；相对attention使用正常、增强正常、伪异常的ArcFace，`s=64,m=28.6°`。最后卷积具体取score conv还是隐藏feature、是否按最大值归一化缺少作者代码验证。c04明确采用隐藏feature通道和、ReLU、逐样本amax归一化，不应称为作者确认细节。

5. **c04不是完全等同论文的记忆寻址。** 正文给的是cosine softmax、hard-shrinkage与L1重新归一化；c04已冻结为dot logits。其slot数和其他工程选择也未获作者确认。本轮名称应始终使用“固定c04独立实现”；保留它作为唯一性能baseline合理，但不能将新结果表述为官方zxVAD复现。

6. **原评分只需G。** 补充材料规定四帧输入、256×256、batch8、5000iterations、逐测试视频PSNR minmax。论文的normalcy模型主要承担训练引导，不能仅因它未直接参与推理就认定设计有误。NC-only与固定融合属于新的预先冻结读出实验，须保留原PSNR主对照及源域校准过程。每视频单调变换保持该视频内部排序，却可能改变跨视频合并frame AUROC。

上述机制及超参数的原始出处：[作者MERL论文版本](https://www.merl.com/publications/docs/TR2023-001.pdf)、[CVF官方补充材料](https://openaccess.thecvf.com/content/WACV2023/supplemental/Aich_Cross-Domain_Video_Anomaly_WACV_2023_supplemental.pdf)。已缓存文本分别位于`research_notes/zxvad_related_works_audit_20261007/papers/zxvad_merl_TR2023-001.txt`与`zxvad_supplement.txt`。当前独立实现的具体事实直接来自冻结`repository/src/baseline_model.py`及`pipeline.py`，不是推测作者私有代码。

## CNG-FP新颖性边界

[Learning Not to Reconstruct Anomalies（BMVC 2021）](https://arxiv.org/abs/2110.09742)已经提出伪异常输入对应干净正常输出的训练目标。其式(3)生成伪异常、式(5)重建对应原正常序列；§3.3.1使用平滑patch及连续位置变化，式(6)定义patch移动，§3.3.2使用skip-frame伪异常。它使用序列重建和同域评测；本轮使用past4→next未来预测及固定cross-domain协议，是应用与任务差异，尚不足以证明新算法贡献。[原论文全文](https://arxiv.org/pdf/2110.09742)。

因此不能把“有连续运动的伪异常+干净目标”作为首创。必要实验是在已冻结c04上判断：异常输入训练是否比普通恢复增强有独立效果，运动有序干预是否优于静态或顺序打乱的匹配干扰，以及是否仍依赖真实历史运动而未退化成背景/平均预测。

## 最小源域诊断

- 冻结同一G/N，使用未参与这些probe拟合的源域正常片段；预先保存片段索引和干预draw。取真实未来Y与G(X)，对两者使用同一供体、mask、box，形成来源×粘贴的2×2。另加入同强度模糊/压缩和recipient自身patch等hard-normal编辑。记录NC score、PSNR、mask内外误差及attention，不凭一个合成分类分数证明真实异常机制。
- 对G引导单独检查`∂L_guide/∂G`非零及N更新对G无梯度；冻结参数与BN模式均写入配置。保持新的源域probe不读取任何目标帧和标签。
- 对反事实输入训练同时保留clean输入、光照扰动、静态patch、随机次序patch、连续轨迹patch；监督目标始终原干净未来。保持batch/step/源片段/参数量相同。若使用额外G前向，必须登记FLOPs与实际时间，不能称等计算预算。
- 测试非退化：正常clean未来误差、最后帧复制基准、反序/静止过去的变化、mask外预测保护及小面积编辑误差。所有阈值与选择规则在目标评测前冻结。

## 协议与证据限制

已看过Ped1/Ped2/Avenue的大量开发结果，本轮不能称三目标完全blind。可以冻结整批实验后统一评测，并称source-only训练、固定参数的开发集筛查；不得在同轮target AUROC后改方案再混称预注册成功。文档“允许单视频自适应”是其建议协议，不是zxVAD已执行的训练或推理规则；本轮没有必要引入测试时参数更新。两套score可作预声明诊断，但主性能依据仍固定原单视频minmax pooled frame AUROC。

STG目标专用β风险已由主分支的`paper_audit.md`从用户提供PDF核实，当前没有重新实施STG。文档其他近年文献仅是背景列表，未逐篇核实；不能沿用其未核实发表状态或把不同数据/监督协议作为本轮严格SOTA排名。

## 本轮实施边界

上述最小诊断列表含扩展建议，本轮只实现来源×粘贴2×2、PHOTO/NOISE/SELF/STATIC/MOVE/SHUFFLE合成panel、干净未来误差/复制末帧/反转早期历史/校准与冻结BN；暂不实施模糊压缩专门诊断、attention定位或mask内外误差，也不因此声称完整证明捷径。训练与probe的具体像素参数以已冻结protocol/spec/edits.py为准。
