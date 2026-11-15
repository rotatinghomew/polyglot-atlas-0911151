# 跨语种对齐与检索口径

不同合作方交付的向量不共享坐标，直接跨语种余弦比较没有意义。跨语种能力只
使用编辑在 `anchors` 中 `confirmed=true` 的词义对应估计线性映射，方法版本为
`cross-lingual-alignment-v1`。未确认或存在争议的对应（包括一词多义、多对一
指向同一目标的记录）不会参与拟合，只进入 `anchor_review` 复核清单，并附上
它们在已接受映射下的观测余弦和拒入原因，编辑可以据此判断是否升级为确认
对应。人工确认记录本身也不保证无误，因此还要通过几何验证才能生效。

## 证据门槛与候选方案

- 每个语种对至少需要 6 条确认对应（`MIN_CONFIRMED_ANCHORS`），低于门槛直接
  拒配并写明“稀疏对应不足以估计线性映射”。
- 同一目标被多个源指向时，每条证据按 `confidence / 入边数` 降权，避免高频
  目标支配拟合。
- 同时拟合两个候选并比较：`orthogonal-procrustes`（正交映射，保持角度结构）
  与 `weighted-least-squares`（带权最小二乘，允许缩放，秩不足即拒）。按留
  家族验证的 recall@1 与平均余弦择优，两个候选的完整报告都保存在方案里。

## 验证：按文章家族整体留一

交叉验证以 `documents.family` 为分组单位，同一文章家族的全部记录整体留在
同一折，翻译对不会同时出现在训练和验证中。每折报告 recall@1、recall@5、
平均/最小留验余弦；再对留验余弦做 400 次 bootstrap，给出 90% 可信区间。
`trust_band` 的 `credible_threshold` 取留验正确匹配余弦的 5% 分位，检索
候选低于该阈值会标记 `credible=false`，提示编辑复核而不是静默当作等价结果。

## 塌缩与热门向量诊断

诊断在**全量源语料**映射后的空间上计算，而不是只看少量锚点：

| 指标 | 含义 | 拒配阈值 |
| --- | --- | ---: |
| `effective_rank_ratio` | 映射后有效秩 / 目标空间有效秩 | < 0.75 |
| `target_coverage` | top-5 邻接覆盖到的目标文档比例 | < 0.60 |
| `hub_entropy_ratio` | 邻接分布熵相对均匀分布的比例 | < 0.75 |
| `recall_at_1` | 留家族验证的首位命中率 | < 0.75 |

只在高频词上拟合的秩不足方案会同时拉低有效秩比、覆盖率和热门熵，即使它在
锚点自身上的余弦很高也会被拒。退化映射的合成场景由
`tests/test_alignment.py::CollapseDetectionTests` 固定。

## 不可跨越的语种

方案的 `reachability` 按已接受语种对做有向连通分析，逐语种列出 `reachable`
与 `unreachable`。没有可靠连接的语种（如本资料中的 `fr`，维度与索引均独立
且无任何确认对应）在检索时返回类型化拒配 `unreachable-language` 与中文
原因，不会回退到直接余弦比较。多跳路径（如 `zh -> en -> ar`）只沿已接受
边 BFS 得到，映射逐跳组合，可信范围取各跳保守包络，不假设各跳误差独立。

## 版本、索引约定与原子切换

- 对齐方案绑定构建时的 `snapshot_id` 与各语种 `index_version`；检索时若生效
  方案的快照指纹与当前快照不一致，返回 409 `snapshot-mismatch`。
- 目标向量的 `index_version` 与对齐声明的目标索引不一致时跳过比较，沿用
  单语检索“同一索引版本才可比较”的约定。
- 对齐写入 SQLite 注册表（`AlignmentRegistry`），保存与生效指针切换在同一
  事务内完成；检索只读激活版本的完整方案，一次查询的所有语种对必然属于同
  一对齐版本，不会一半用旧空间、一半用新空间。旧版本不可变、可回溯，候选
  结果中带有 `alignment_id`、`alignment_version`、`alignment_path` 与
  `trust_band`。
- 权限仍以 `permissions.allowed_languages` 与 `allowed_collections` 为准；
  无权限返回 `forbidden` / `forbidden-language`。
