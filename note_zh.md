# HAMGF

Hierarchical Agent Memory Graph Framework（分层智能体记忆图框架）使用 NetworkX `MultiDiGraph` 将长期记忆组织为可追溯的因果、时序与语义链。

**测试结果过多而且都存储在tests/sol里面，旧的新的混在一起，懒得选择性挑出来打包:D**

## 推理对话

先确保模型服务和前端依赖已准备好，然后运行：

```bash
cd frontend && npm ci && cd ..
HAMGF_LLM_MODEL='模型名称' PYTHONPATH=src python -m hamgf.chat
```

程序自动启动本地 HAMGF API 与 Vite 图页面、尝试打开浏览器，并在每次成功回答后把“用户问题 → Agent 回答”同步写入 `data/snapshots/phase4_chat.json`。默认模型端点为 `http://127.0.0.1:11434/v1`；其他兼容端点可通过 `HAMGF_LLM_BASE_URL` 配置。

## 独立启动 API 与图页

启动带标准分叉/修正链数据的内存 API：

```bash
PYTHONPATH=src python -m hamgf.api --demo
```

另一个终端启动前端：

```bash
cd frontend
npm ci
npm run dev
```

Python SDK 接入：

```python
from hamgf import HamgfClient

client = HamgfClient("http://127.0.0.1:8000")
client.write_memory("客户确认本周采用方案 B", importance=0.9, timeliness=0.9)
chain = client.search("方案 B", k=5)
print(chain["node_ids"])
```

企业聊天接入会先自动探查每份文件；低置信格式可人工确认：

```python
from hamgf import ChainMemoryGraph, EnterpriseMemoryAdapter, LifecycleMemoryWriter

graph = ChainMemoryGraph()
adapter = EnterpriseMemoryAdapter(LifecycleMemoryWriter(graph))
result = adapter.insert_file("data/raw/deidentified-chat.txt")
print(result.report.to_dict(), result.node_ids)
```

15 份真实脱敏记录的聚合案例验证可复现运行：

```bash
PYTHONPATH=src:. python scripts/validate_enterprise_cases.py
```

当前结果为 15/15 案例通过、980 条事件、867 个 CMG 节点、852 条逻辑边，报告和论文图见论文（没上传sol文件夹导致的）。

## Neo4j 与容器部署

可选持久化依赖不会进入核心安装：

```bash
pip install -e '.[persistence,security]'
HAMGF_NEO4J_URI=bolt://127.0.0.1:7687 \
HAMGF_NEO4J_USER=neo4j HAMGF_NEO4J_PASSWORD='替换为本地密码' \
PYTHONPATH=src python -m hamgf.api --neo4j
```

`Neo4jGraphStore` 按 namespace 保存经 schema 校验的节点和多重边，可从 Neo4j 重建 `ChainMemoryGraph`；真实 Neo4j 5.26.30 的 7 节点/7 边往返哈希一致。

生成 AES-256-GCM 密钥后可启用加密快照；密钥只通过环境变量传递：

```bash
PYTHONPATH=src python -c 'from hamgf.persistence import EncryptedSnapshotStore; print(EncryptedSnapshotStore.generate_key())'
HAMGF_SNAPSHOT_KEY='替换为上一步输出' PYTHONPATH=src python -m hamgf.api \
  --snapshot data/snapshots/runtime.enc --encrypted-snapshot
```

如需保护 REST 接口，再设置独立的高熵 Token。`/health` 保持公开，所有 `/v1/*` 受保护：

```bash
HAMGF_API_TOKEN='替换为高熵Token' PYTHONPATH=src python -m hamgf.api --require-auth
```

SDK 使用 `HamgfClient(base_url, api_token=...)`。

安装 Docker 的机器可运行：

```bash
HAMGF_NEO4J_PASSWORD='仅本机部署密码' \
HAMGF_SNAPSHOT_KEY='base64 AES-256密钥' docker compose up --build
```

图页与 API 默认只映射到 `127.0.0.1:8080` 和 `127.0.0.1:8000`。Neo4j 数据卷本身仍需由宿主机磁盘加密保护。

## 固定模型 benchmark（历史 n=20 协议，不作为当前操作入口）

主实验使用Qwen3.6-27B、DeepSeek V4-Flash 与 Gemini 3.1 Flash-Lite。新版正式基线为 No Memory、FullText、HybridRAG、完整 Mem0 OSS、MemOS 与 HAMGF。Naive RAG被替换；Graphiti被我踢了。框架真实抽取和检索得到的证据在三模型间冻结复用；统一中性提示词、3000 个cl100k_base 证据 tokens 和生成预算。FullText 是不截断完整历史的容量参考组。建库、检索、纯模型响应、在线总耗时、含建库总耗时分别记录。其他信息见 [基线说明](docs/memory-baselines.md)。

历史 n=20 同样本离线回放的准确率如下（每模型、每策略各 20 例）：

| 记忆策略 | Qwen3.6-27B | DeepSeek V4-Flash | Gemini 3.1 Flash-Lite |
|---|---:|---:|---:|
| No Memory | 0% | 0% | 0% |
| FullText | 90% | 55% | 70% |
| HybridRAG | 80% | 65% | 70% |
| Mem0 OSS | 0% | 0% | 0% |
| Graphiti | 20% | 10% | 20% |
| MemOS | 70% | 50% | 65% |
| HAMGF（完整生命周期） | 90% | 60% | 70% |

HAMGF 的平均纯生成时间分别为 3.824 / 1.277 / 1.719 s；三模型描述性合计为 44/60， FullText 与 HybridRAG 均为 43/60。完整准确率、95% CI、时延、逐案例审计与旧 PDF/SVG/PNG 与生命周期报告已随早期结果清理，不再提供失效链接；现行结果入口见 [评测说明](docs/evaluation-and-validation.md)。 20 案例共真实录入 127 条历史，检测 6 对真实修正冲突并记录 12 次双端降权，20 条 buffer 控制记录全部 TTL 遗忘，20/20 案例完成链检索和分层 HCI。本轮 HAMGF 未稳定优于最强基线，不能据此宣告普遍优越。样本只有 20 个离线任务，不是官方交互式成绩；No Memory 是证据不足时拒答的对照，FullText 是不截断容量参考， Mem0 原生个人信息抽取在 19/20 搜索案例中为空，不能据此推断其个人记忆能力。

旧简化 v2 的 HAMGF 失败分层诊断（历史报告已归档）显示，19 个错误全部为主动拒答，且答案字符串在 19 个实际提示词中都可见；11 个错误也被 FullText、HybridRAG、MemOS 同时拒答，不能全部归因于图召回遗漏。独立 `detail_once_v1` 配对消融（历史报告已归档）只删除摘要与详情的重复副本，结果为 Qwen 95%、DeepSeek 40%、Gemini 65%，相对默认的 85%/60%/60% 合计净减 1 个正确案例。因此默认 HCI 未改变，该模式仅保留作负消融。

旧合成冒烟命令：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src \
python scripts/run_configured_benchmarks.py
```

原生 v2 的 n=20 复现命令如下（当前 32768 独立版使用上方链接中的命令）：

```bash
python scripts/setup_memory_baselines.py
python scripts/baseline_neo4j.py install
python scripts/baseline_neo4j.py start
PYTHONPATH=src:. python scripts/prepare_memory_baselines.py --local-neo4j
PYTHONPATH=src:. python scripts/run_memoryarena_benchmark.py --limit 20
PYTHONPATH=src:. python scripts/run_memoryarena_benchmark.py --generator deepseek --limit 20
PYTHONPATH=src:. python scripts/run_memoryarena_benchmark.py --generator gemini --limit 20
PYTHONPATH=src:. python scripts/summarize_memoryarena_primary.py
```

三池四象限复核运行 python scripts/run_classifier_review.py。JSON 与 Matplotlib 图表统一输出到 tests/sol/，包含矢量 PDF/SVG 和 300 DPI PNG。详情及当前实测限制见 `docs/evaluation-and-validation.md`。

单进程性能与压力测试：

```bash
PYTHONPATH=src:. python scripts/run_performance_benchmark.py
```

当前 250/1000/2500 节点档位全部通过回归门槛；2500 节点链式检索 P95 为 406.05 ms，峰值内存 30.45 MiB，加密快照保存/恢复为 180.72/358.02 ms。8 worker 的 32 次认证 REST 检索全部成功，P95 为 882.12 ms。详情与 PDF/SVG/PNG 见 `tests/sol/performance/`。输出会记录实际操作系统、Python、容器和 CI 环境；结果不代表分布式生产 SLA，也不覆盖 Neo4j 网络吞吐或前端大图渲染。

个人/Coding 场景使用 GitHub REST API 获取并最小化保存 NetworkX 公共历史：

```bash
PYTHONPATH=src:. python scripts/fetch_coding_history.py
PYTHONPATH=src:. python scripts/validate_personal_scenarios.py
```

固定快照包含 20 条 Commit 与 20 条非 PR Issue，提交邮箱、Issue 正文、头像等无关字段不落盘；40 条记录全部进入独立 CMG，目标 Issue 在 8 节点连贯链中命中，加密恢复哈希一致。真实本地 Llama-3.2-3B 在无记忆时未回答出合成项目代号，接入 6 轮 HAMGF 后准确召回并显式引用 6 节点链。原始模型回答只保留 SHA-256，不写入报告。完整结果与 PDF/SVG/PNG 见 `tests/sol/personal-validation/`。这是一项功能性案例验证，不代表通用模型准确率或统计显著性。

MemoryArena 的首个公开数据接入固定为 progressive_search 官方版本，可复现获取并生成离线回放输入：

~~~bash
PYTHONPATH=src python scripts/fetch_memoryarena.py
~~~

脚本校验版本与 SHA-256，原始/处理数据分别写入 data/raw/memoryarena/ 和 data/processed/memoryarena/，schema 审计及 PDF/SVG/PNG 数据概况图写入 tests/sol/memoryarena-progressive-search/。离线回放输入不等同于官方交互式环境成绩。

公开数据集的 v4 实验由 `scripts/run_phase4_public_dataset_pipeline.py` 运行，依次准备基线、构建 HAMGF v4 计划、运行模型并汇总结果。准备好对应数据后，可先查看命令：

~~~bash
python scripts/run_phase4_public_dataset_pipeline.py --profiles locomo msc --dry-run
~~~

移除 `--dry-run` 后执行。该入口还支持 `bundled_shopping` 和 `group_travel_planner`； LongMemEval 和 PrefEval 分别使用 `scripts/run_longmemeval_n50_pipeline.py` 和 `scripts/run_prefeval_pipeline.py`。

runner 会先做 judge 正反例校准；答案生成后立即保存独立断点，judge 网络中断不重生成答案，判分完成再写入完整 checkpoint。传输重试另存事件日志；生成时间与 judge 时间分开统计，正确率同时给出 95% Wilson 区间。

**历史四策略结果（不代表当前 v4 成绩）**：HAMGF 在同一批 20 个任务上分别达到 Qwen 19/20、DeepSeek 19/20、Gemini 17/20；三模型汇总报告位于 tests/sol/memoryarena-progressive-search/primary-models-n20/。这仍是离线回放，不能替代官方交互式成绩。

## 验证

```bash
python -m tests
```

该命令同时运行 Python、前端 Node 测试与生产构建，结果统一导出至 `tests/sol/`，包括 TXT、JSON、JUnit XML、SVG、前端 TAP/构建日志及历史趋势数据。

详细说明见 [`docs/architecture.md`](docs/architecture.md)、 [`docs/memory-lifecycle.md`](docs/memory-lifecycle.md)、 [`docs/visualization-and-integration.md`](docs/visualization-and-integration.md) 与 [`docs/evaluation-and-validation.md`](docs/evaluation-and-validation.md)。
