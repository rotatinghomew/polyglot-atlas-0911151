# 业务资料契约

vectors 的 values 仅在同 language 和 index_version 内共享坐标，语种之间的维度也可以不同。documents.family 关联同一素材家族及翻译，collection 决定可访问资料范围。anchors 是词义对应候选，confidence 和 confirmed 是人工记录，不保证无误，也不保证每种语言连通。跨语种对齐（见 `docs/alignment.md`）只把 confirmed=true 的记录作为拟合证据，未确认或多对一的记录仅进复核清单；family 仅用于按家族整体留一的验证，不参与拟合。permissions 是检索端权限；当前禁用材料还受 policy 约束。

## 资料表

| 表名 | 原始记录数 | 字段 |
| --- | ---: | --- |
| vectors | 188 | id, index_version, language, released_at, revision, values |
| documents | 188 | collection, family, id, language, released_at, revision, text, title |
| anchors | 27 | confidence, confirmed, id, left, released_at, revision, right, sense |
| permissions | 1 | allowed_collections, allowed_languages, id, released_at, revision |

记录均为本地合成资料，只表达业务结构，不是公开报道所述人群或机构的实测资料。日期是本地资料日历；所有版本以资料服务的 as_of 为准。字段值和自然缺失保留原义。金额、评分、频率和权重不可跨表不加判断地直接合计。
