# HAMGF 架构与接口

## 总体架构

HAMGF 以 `networkx.MultiDiGraph` 为核心，运行时由 `MemoryApplication` 统一持有图、池管理器和生命周期写入器。REST API、Python SDK、数据适配器与推理 Agent 都建立在这一运行时之上。

```mermaid
flowchart TD
    sources["数据集 / 企业聊天 / Coding 记录 / REST / Python SDK"] --> app["MemoryApplication<br/>线程锁、revision、增量事件、持久化协调"]
    app --> writer["LifecycleMemoryWriter"]
    writer --> validation["分类与可信度验证"]
    writer --> pools["MemoryPoolManager<br/>buffer/archive + hot/warm/cold"]
    validation --> graph["ChainMemoryGraph<br/>MultiDiGraph"]
    pools --> graph
    graph --> search["ChainSearch"]
    search --> hci["HCI"]
    hci --> agent["MemoryGroundedAgent"]
    agent --> llm["LLM"]
    graph --> persistence["JSON / AES-256-GCM / Neo4j（可选）"]
```

核心层只要求轻量 Python 依赖与 NetworkX。LLM、Neo4j、加密和前端均通过可选边界接入，不是构造 CMG 的前置条件。

## 三层职责

### 1. 获取与维护层

相关模块：`hamgf.ingestion`、`hamgf.pools`。

- `MemoryClassifier` 根据重要性、时效性与可配置阈值产生 `working / episodic / buffer / archive` 分类。概念上仍是工作、情景长期、归档三个逻辑池，因此还是称之为三池分类。`buffer` 是低重要、高时效记录使用的过渡子池。
- `CredibilityEngine` 提供来源初始评分、锚点可信度继承、指数衰减、多源增信、冲突降权和用户重申能力。默认写入路径会执行初始评分、继承与冲突处理。
- `LifecycleMemoryWriter` 在四步写入协议之后进行池路由：`working` 与 `episodic` 节点进入 CMG，首次分类为 `buffer` 或 `archive` 的记录会进入独立池。
- `MemoryPoolManager` 管理 buffer TTL、归档记录、引用晋升、池流转审计以及 hot/warm/cold 分层。已经在 CMG 中的节点归档时采用软删除，仍保留在图内；压缩产生的归档快照节点也会保留在 CMG，以保证可审计性。
- `DynamicMemoryCompressor` 实现节点/边注意力排序、TTL 清理、冷热重平衡、久远分支快照压缩、低注意力节点归档和分支上限控制。压缩器实现 `CompressionPolicy` 协议，但维护操作需要显式调用，不会由 `MemoryApplication` 自动定时运行。

可信度变更历史当前保存在 `CredibilityEngine` 的进程内审计日志中，可通过 API 图视图和节点视图读取，进程重启后不会自动恢复。归档保留原节点内容/向量并新增摘要节点。

### 2. 表示与组织层

相关模块：`hamgf.core`

- `MemoryNode` 和 `MemoryEdge` 是冻结的数据类 schema；创建与更新时 校验枚举、 `[0,1]` 分数、时间戳、有限数值向量和 JSON-compatible metadata。
- `ChainMemoryGraph` 封装 `networkx.MultiDiGraph`，支持节点/边 CRUD、分叉、回流、快照导入导出和图不变量校验。
- 边的标准关系始终为 `causal / temporal / semantic`。修正链使用“新节点 → 被推翻旧节点”的 `semantic` 边，标签为 `supersedes`；旧节点保留并标记为 `superseded`。
- `MemoryWriter.write()` 编排四步写入协议：定位锚点、建立逻辑边、可信度继承、冲突时局部重构。
- `nx_graph` 返回冻结副本，只用于 NetworkX 读算法。

### 3. 调用与推理增强层

相关模块：`hamgf.retrieval`、`hamgf.agent`。

- `ChainSearch` 先为活动候选节点评分并选择一个最佳入口，再按受限 beam traversal 向前溯因、向后延果，返回最多 `k` 个相连节点组成的有序 `ChainResult`。
- 检索默认使用轻量词法相似度。当节点已存 embedding 且调用方同时传入同维度的 `query_embedding` 时，检索器会组合余弦相似度。向量缺失或不可比较时回退到词法分数。
- `build_context()` 提供摘要/全文两级 `HierarchicalContext`。当前推理运行时使用 `build_hci_prompt()` 生成有字符上限的上下文，
   包含节点 ID、类型、池、状态、可信度、摘要、详情和边轨迹。
- `MemoryGroundedAgent.ask()` 执行“检索 → HCI → LLM → 链引用包装”，并在启用 `record_conversation` 时把当前轮用户输入和 Agent 回答同步写回 CMG。

## 应用、API 与插件边界

### 适配器与公共库的区别

benchmarks/hamgf_v4.py 复用生命周期与图组件，以 512/64 token 事件块、外部 embedding、BM25+dense+RRF 至多三个入口和门控邻接扩展构造最多 k=6 的证据子图；允许多个弱连通分量。完整配置与实际触发次数见[评测说明](evaluation-and-validation.md)。

### `MemoryApplication`

- 持有同一份 `ChainMemoryGraph`、`MemoryPoolManager` 和 `LifecycleMemoryWriter`
- 以可重入锁保护公开运行时操作
- 提供写入、检索、图视图、节点详情、审计、边修正和快照操作
- 维护当前进程内的 revision 与增量事件，供前端轮询
- 在配置后执行图快照自动保存或 Neo4j 状态同步

revision和增量事件列表属于进程内状态，重启后从零开始。

### REST、SDK 与前端

- REST 服务提供 `/health`、`/v1/graph`、`/v1/snapshot`、`/v1/events`、 `/v1/nodes/{id}`、`/v1/audit`、`/v1/memories`、`/v1/search` 和 `/v1/edges`。
- `/v1/*` 可选 Bearer Token 认证；CORS origin 可配置。`/health` 保持公开以供探活。
- `HamgfClient` 是无第三方 HTTP 依赖的 Python SDK，对上述端点提供薄封装。
- React 前端通过图快照和增量事件渲染横向链、分叉、回流、状态与可信度详情。
- 企业聊天、个人本地模型、Coding 历史及不同 LLM 后端通过 `hamgf.adapters` 接入。

## 持久化边界

| 方式 | 保存内容 | 不保存的运行时内容 |
|---|---|---|
| NetworkX JSON 快照 | CMG 节点、边和图 schema | 独立 buffer/archive 记录、冷热层、池流转、可信度历史、revision/增量事件 |
| AES-256-GCM 加密快照 | 与 JSON 快照相同，但认证加密并原子替换 | 与 JSON 快照相同 |
| Neo4j `push_state/pull_state` | CMG + buffer/archive + 冷热层 + 池流转状态 | 可信度历史、revision/增量事件 |

Neo4j 是可选的大规模持久化后端。服务以 `--neo4j` 启动时从其恢复图和池状态，后续变更同步回后端。

## 主要 Python 接口

建议从应用层入口完成日常写入和检索

```python
from hamgf import MemoryApplication

app = MemoryApplication(
    snapshot_path="data/snapshots/cmg.json",
    autosave=True,
)

written = app.write_memory(
    {
        "content": "客户确认采用修订方案 B2",
        "summary": "确认方案 B2",
        "type": "feedback",
        "importance": 0.95,
        "timeliness": 0.9,
        "source": "user_confirmed",
    }
)
chain = app.search({"query": "最终采用了哪个方案？", "k": 5})

print(written["node"]["node_id"])
print(chain["node_ids"])
```

维护任务需要显式调用，并在直接修改图后显式保存：

```python
from hamgf import DynamicMemoryCompressor

compressor = DynamicMemoryCompressor(pool_manager=app.pool_manager)
report = compressor.apply(app.graph)
app.save_snapshot()

print(report.archived_nodes)
print(report.snapshot_nodes)
```

若需要推理，由调用方构造符合 `LLMBackend` 协议的后端，再交给 `MemoryGroundedAgent`。

## 自动执行与显式操作

| 行为 | 默认写入/推理路径 |
|---|---|
| 分类、来源可信度、锚点继承、冲突处理、池路由 | 自动 |
| 配置后的图快照 autosave、Neo4j 同步 | 自动 |
| 链式检索、HCI、链引用、对话回写 | 使用 `MemoryGroundedAgent.ask()` 时自动 |
| 可信度时间衰减、多源增信、用户重申 | 显式调用 |
| TTL 清理、注意力重算、冷热重平衡、压缩/遗忘 | 显式调用 `DynamicMemoryCompressor.apply()` |
| embedding 生成 | 由调用方或适配器完成 |

