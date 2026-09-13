在项目根目录运行：

```bash
python -m tests
```

每次运行都会覆盖最新报告，并在 `history.jsonl` 追加一条历史摘要

- `latest.txt`：Python `unittest` 详细文本输出；
- `latest.json`：机器可读的运行环境、汇总、套件和逐项结果；
- `junit.xml`：供 CI 与测试平台导入的 JUnit XML；
- `summary.svg`：可直接预览的静态结果图；
- `history.jsonl`：跨运行的趋势数据；
- `frontend-test.tap`：前端 Node 测试的原始 TAP 输出；
- `frontend-build.txt`：React Demo 与可嵌入插件的生产构建日志。
- `reasoning-smoke/`：确定性推理管线的对照结果；
- `model-benchmark/<model>/`：其他模型在同一小型 benchmark 上的补充结果；
- `model-benchmark/suite-summary.{pdf,svg,png}`：跨模型提升与响应差补充图；
- `classification-review/`：24 条标注样本的三池四象限复核、混淆矩阵图。
- `neo4j-sync/`：真实 Neo4j 的 NetworkX 双向往返哈希图。
- `security-validation/`：AES-GCM 快照与 Bearer 认证的 9 项控制及可视化审计。
- `performance/`：250–2500 节点核心扩展、加密快照和并发 REST 压力结果。
- `neo4j-runtime-state-sync/`：主图、buffer、archive 与池状态的真实 Neo4j 无损往返。
- `neo4j-network-performance/`：250–2500 节点真实 Bolt 吞吐与隔离并发全状态验证。
- `frontend-large-graph/`：1k/5k/10k 节点生产转换与 headless Cytoscape 布局/增量更新。
- `phase4-expanded-pipeline-tests/`：公开数据集流水线测试，覆盖断点恢复、模型配置和首字响应时间统计。
- `phase4-api-preflight/`：真实一 token 抽取端点可用性结果；余额/认证失败会在框架 worker 启动前落盘。
- `phase4-api-preflight-tests/`：预检最小 token、HTTP 402 结构化、密钥不落盘与原子写入回归。
- `reference-eval-tests/`：reference report 数据集身份持久化、LongMemEval 标题与 SVG 离线重绘回归。
- `locomo-n50/`、`msc-session5-n50/`、`memoryarena-bundled-shopping/`、`memoryarena-group-travel-planner/`：扩展公开 replay 的固定输入、manifest、协议审计及可视化。
缺少 Node.js/npm 或未执行 `frontend/npm ci` 时，前端测试明确标记为跳过，其余 Python 测试仍可运行。
