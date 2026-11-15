# 按语种分离的本地向量索引与素材权限检索

polyglot-atlas 是 Python 业务分析应用，目前提供原始资料核对和已有口径概览；摘要只说明观测记录本身，不代表尚未建立的推断、个性化、潜在分布或决策收益。

## 目录和边界

`polyglot_atlas/data.py` 读取资料并冻结公开日期可见的版本。`product.py` 实现当前领域摘要与单语检索，`alignment.py` 只用编辑确认的词义对应构建跨语种空间对齐（含按文章家族整体留一的验证、可信范围与塌缩诊断），`registry.py` 保存不可变对齐版本并原子切换生效指针，`crosslingual.py` 在生效版本上执行跨语种检索，`workflow.py` 串联资料读取与成果发布，`service.py` 提供本地 HTTP 资料和消费接口，`store.py` 保存不可变成果和当前消费约束，`client.py` 是调用适配层。服务接口见 `docs/protocol.md`，具体资料含义见 `docs/data-contract.md`，跨语种口径见 `docs/alignment.md`。`tests` 覆盖已有业务口径与联调，不代表新增方法已经实现。

## 环境和依赖

兼容 Python 3.11.x，具体范围见 `pyproject.toml`；`.python-version` 固定受支持的次版本，不限定补丁版本。独立 Docker 镜像使用固定摘要的 Python 3.11.14，生产验证容器使用 Python 3.11.2，两者采用相同的完整锁定依赖。`requirements.lock` 与 `requirements.txt` 固定数值计算和模型分析的完整依赖版本。应用运行、测试及资料均在本地，不请求外部 API；构建时预装依赖，正式运行不安装或下载资料。

```bash
docker build -t polyglot-atlas:local .
docker run --rm --network none polyglot-atlas:local python tools/demo.py
docker run --rm --network none polyglot-atlas:local python -m unittest discover -s tests -v
```

已有 Python 环境可以先执行 `python -m pip install -r requirements.lock`，然后执行以下检查与产品演示：

```bash
python -m compileall -q polyglot_atlas tests tools
python -m unittest discover -s tests -v
python -m build --wheel --no-isolation
python tools/demo.py
```

演示会启动真正的本地 HTTP 服务，读取业务资料，计算当前概览、发布后读回成果，结束时自动退出。每次使用临时数据库，不覆盖工作目录。它展示当前产品，新增分析以最终实现的业务入口为准。

## 交互服务

```bash
python -m polyglot_atlas.service --port 8765
```

默认服务地址为 `127.0.0.1:8765`，本地成果保存在 `local-state.sqlite3`，对齐版本保存在 `local-alignments.sqlite3`。启动后资料只读；本地消费约束可通过管理接口变更，不会修改已发布成果。中断运行可正常退出。已有概览使用 `observed-overview-v1` 标明方法版本，跨语种对齐使用 `cross-lingual-alignment-v1`，额外分析可使用自己的方法和结果结构，不能复用旧方法名掩盖语义变化。

## 跨语种检索

先 `POST /alignments` 构建并原子激活对齐方案，再调用 `GET /search/cross-lingual`。没有可靠连接的语种（如无任何确认对应的 `fr`）会带 `unreachable-language` 原因明确拒配，不会回退为直接余弦比较；每条候选都带依赖的对齐版本、对齐路径、可信阈值与可信标记。更新对齐在单个事务内切换生效指针，查询不会混用新旧空间。完整口径见 `docs/alignment.md`。
