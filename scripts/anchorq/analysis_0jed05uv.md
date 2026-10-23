# AnchorQ run 0jed05uv 审计与 v2 修正

审计日期：2026-10-23。Run：https://forge.coreweave.com/wandb/yuwanliu06/OGPO/runs/0jed05uv

通过已认证 W&B API 取得配置、summary 和 evaluation history，并与本地 CSV 对齐。
原始快照在 `run_logs/analysis-0jed05uv/`；该目录通常被 git 忽略。

## 观察到的事实

- W&B 初次抓取时在线约 4.9 万步，前四次评估（在线 1/2/3/4 万步）成功率均 0，return 均 -400。
- 审计结束前本地 CSV 新增在线 5 万步评估，仍为 0/-400。
- 日志横轴包含 90,000 offline offset：例如 step=130000 是 online=40000。
- 在线 15k–30k，Q 从约 -99.913 到 -99.993，TD loss 很小。持续 -1 奖励、gamma=.99 时失败固定点是 -100；因此 loss 小不意味着策略变好。
- 粒子 warmup 是 10,000 **决策**。horizon=4 时，前三次评估尚未启用粒子；第四次评估刚到启用边界，缺少启用后的学习时间。5 万步仍失败，不能只用 warmup 解释。
- 同期本地 baseline 评估如下；DSRL/GRACE 的 10k/20k 可能相差 1 个物理步。这里只比较同阶段，不是最终成绩或多 seed 统计。

| 在线步数 | AnchorQ 旧版 | OGPO | DSRL | GRACE |
|---|---:|---:|---:|---:|
| 10k | 0 | .28125 | .1875 | .21875 |
| 20k | 0 | .359375 | .09375 | .03125 |
| 30k | 0 | .328125 | 0 | .015625 |
| 40k | 0 | .3125 | .078125 | 0 |

## 复现偏差及修正

旧版保留核心 anchor / reversal / particle barycenter 规则，但将 Appendix B 的随机先验去掉、PER 换成 uniform、REDQ min target 换成 mean。它是方法简化版，不能作为完整论文复现。为匹配 baseline 而删除这些方法组件不合适。

v2 已修正：

1. 每个 Q head 使用 `f_i(s,z) + 2 * p_i(s,z)`。p 是独立初始化且冻结的同结构 Value 网络；无 optimizer。选择、粒子权重、ensemble uncertainty、online/target Q 全部包含同一个冻结 prior。
2. TD bootstrap 使用随机抽取两个 target heads 的最小值；下一候选由 online ensemble mean 选择。
3. PER 使用 sum/min tree，重要性权重进入 TD loss，每个 optimizer step 后立即更新该批 priority，再抽下一批。
4. 新日志包括候选 Q 差距、head disagreement、粒子是否启用、latent RMS、执行动作的夹爪绝对值和成功 replay 数量。
5. 实验名改为 `anchorq_square_shared_ogpo_bc_v2`，不覆盖旧版目录。

PER alpha=.6、beta=.4→1、epsilon=1e-6、跨 head 的 mean absolute TD priority 是明确的实现选择，论文没有给出这些完整细节，不宣称是原作者参数。

## 公平比较的边界

共享 `ogpo_squareseed1/params_90000.pkl`，相同的 decoder 与可训练 Value 4×512 / 10 heads、batch256、lr4.5e-5、gamma.99、tau.05、horizon4、在线 2M 物理步和两次 critic 更新预算。没有添加 offline Q 数据、成功奖励、额外可训练 actor 或训练步数。

随机先验增加冻结参数和前向计算量，PER 改变采样分布，这是恢复方法组件的代价，不意味着总计算量相同。保留 baseline 的额外成功轨迹 Q 更新（成功子集内 uniform、权重1，同时刷新其 PER priority），以及物理步 discount/UTD，仍属于公平比较所需的 benchmark 适配。Robomimic 小型 flow 与论文 VLA/LIBERO 不同，不能保证原文的相对收益迁移。

## 反解和参考动作检查

真实 checkpoint 上，AnchorQ reverse Euler 与 GRACE inverse_fm_batch 的普通 batch 及六参考动作 batch 最大绝对误差均为 **0.0**。这排除了“两种反解公式不一致”，不代表有限步 Euler 是精确逆映射。

在 square/mh 数据前16条 demo 各抽首帧和中间帧，共32个状态：六方向 latent RMS 约1.34–1.68（Gaussian defer 约1.03），最大单维绝对值最高13.16。六方向解码后的夹爪绝对值均值约.59–.66，defer 约.994，demo 为1.0。这说明固定方向参考会明显改变原 BC 动作分布；它可能降低早期有效抓取，但这些样本不是在线因果实验，不能据此断言是唯一根因。

FRS 原文附录 D.2 也使用零旋转/零夹爪参考，并依赖 flow 解码产生相应动作，因此没有擅自把夹爪改成强制闭合或改成新的 BC-guided reference 来“追分”。

## 验证与后续判断

CPU 回归覆盖真实 checkpoint、反解、粒子公式、prior 冻结、PER 分布/IS、终止与截断 bootstrap、成功 buffer 及实际 optimizer 计数。另有 DSRL checkpoint/budget 回归。没有完成 v2 全量 Robomimic 训练，不声称成功率已恢复。

重跑命令见同目录 README。旧进程不会热更新，本次没有停止旧进程或启动新的长训练。应使用相同 seed/预算重跑 v2，并关注 40k 之后成功 replay 是否出现、ensemble disagreement 与 candidate Q spread 是否持续有区分度。最终结论需要完整训练和多个 seed；不能仅依据 loss 或前几次零成功评估。
