# V1公开提交的实现失效与V2修复

原始提交：61f9232ea11e5238ed65b7dfa7fc1636f457f3c1，分支c04-mechanism-audit-3090-20261009。新修复版只改变正确更新和验证门，不改变研究假设、32组清单、seed17、5000step、batch8、297/33划分、抽样计划、编辑参数、源选择规则或读出公式。

V1的experiment_update.py在计算ln后遗漏opts['N'].zero_grad、ln.backward和opts['N'].step；该优化器包含N和Arc。因此30个非baseline的N/Arc参数未学习。B0/B1委托原始更新函数，2项有效。旧检查覆盖梯度和BN次数，却未证明参数变化/Adam步数，这是生成代码和验证设计的遗漏。

553份导出成员SHA和672项原始AUROC重算一致，但不使失效方法比较有效。多个NC样本来源方案的逐帧PSNR完全一致是具体症状；不能据此否定NC或CF，也不能用目标结果挑选方案。模型权重未上传，checkpoint SHA只能核对记录，未独立重放模型。

V2补回N/Arc反向与Adam更新；每个正式step验证G/D/N全部Adam参数状态和更新计数，预检查额外验证N/Arc参数变化。CPU32×3步和24CF p=0×3步原始更新全状态比较；在每张卡上，用真实297训练计划前8条、真实模型进行三个临时源批次更新，比较原始c04与新分支p=0的全部模型/Adam/Torch RNG以及原始指标。临时模型丢弃、不写checkpoint、不接触目标。receipt绑定release、prepare、plan、split和物理卡，SHA进入preflight、freeze、completion及公开导出。CPU通过不等于CUDA通过；服务器GPU检查未过时不得开始正式训练。

新目录/home/xjy/zxvad-c04-mechanism-audit-v2，32组全部从step0开始，不加载V1权重。B0/B1虽原始有效仍重跑，保证新发布版本、同卡、同协议配对。原V1公开证据与本地快照保留并标注INVALID_IMPLEMENTATION，不改写，不与V2混汇。

V1实际墙钟6h44m包括缺少N反向的训练，不能直接当修复版预算。暂估双3090约8–12小时（非保证，依实测吞吐更新），无截止时间、无多种子或新增参数搜索。完成后才审阅AUROC；所有负结果一并公开。
