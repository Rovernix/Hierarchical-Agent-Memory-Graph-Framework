# 基线协议

本协议比较**同一回答模型下的记忆框架**。Qwen3.6-27B、DeepSeek V4-Flash、Gemini 3.1 Flash-Lite 为主要生成模型，GPT-5.5 为 reference judge。

| 组别 | 实际实现 | 边界 |
|---|---|---|
| No Memory | 无历史证据 | 与其他组使用同一 reader 提示词 |
| FullText | 全部过去历史，原始顺序 | 无证据截断；容量参考组，不视为等 token 检索组 |
| HybridRAG | BM25 + text-embedding-3-small + RRF | 512-token 分块，重叠 64；不是旧纯向量 Top-k 的改名 |
| Mem0 Full OSS | mem0ai 2.0.14 + Qdrant local | `infer=True`、spaCy 实体、fastembed BM25、原生混合评分；不使用可选额外 cross-encoder 或 Mem0g 图模块，不等同托管平台 |
| MemOS | MemoryOS 2.0.33 + Neo4j | SimpleStructMemReader fine 抽取 + TreeTextMemory 同步重组 + BM25；文本记忆路径，不启用 KV/LoRA 参数记忆或私有微调模型 |
| HAMGF | NetworkX CMG + full lifecycle + chain search + HCI | v4 使用事件节点、四标签路由、可信度、冲突与维护调用、三类边、BM25+dense+RRF 多入口和门控扩展；机制调用不等于每例触发压缩 |

## PrefEval

[PrefEval](https://github.com/amazon-science/PrefEval) 用于比较不同记忆方式能否在长对话中保留用户偏好。本项目使用 revision `50795054b5ff5f418d2b768a331d71e480f93331`；原始数据及衍生数据受 CC BY-NC 4.0 许可约束。

默认配置从 explicit、choice-based 和 persona-driven 三种偏好形式中选择 50 个任务，seed 为 `20260909`，并在最终请求前插入 10 轮无关对话。各方案接收相同的历史和最终问题；参考偏好与解释只交给 evaluator，不进入记忆系统或回答模型。explicit 形式中的原始确认语会替换为中性确认语，避免确认内容提前泄露偏好信息。

evaluator 判断回答是否识别偏好、虚构偏好、违反偏好，以及是否对最终请求有帮助，同时记录准备、检索、生成和评估耗时。由于本项目改变了执行和评分流程，结果仅适合在相同数据、模型配置和方案集合下横向比较。

从项目根目录运行以下命令：

```bash
python scripts/fetch_prefeval.py
python scripts/run_prefeval_pipeline.py
```

第二条命令只显示执行计划。配置模型端点、API key 环境变量和本地模型路径后，使用下面的命令运行：

```bash
python scripts/run_prefeval_pipeline.py --execute
```

已有的 PrefEval 本地仓库可以通过 `python scripts/fetch_prefeval.py --source-tree /path/to/PrefEval` 读取。`--start-at` 和 `--stop-after` 可限制执行阶段；完整参数以 `python scripts/run_prefeval_pipeline.py --help` 为准。默认输出位于 `tests/sol/prefeval-n50-t10/`

## n=50 主实验

论文主准确率矩阵只纳入当前协议下可排名的四组任务：

LoCoMo、LongMemEval-S、MemoryArena `progressive_search` 与 PrefEval controlled replay。

每组固定 50 个案例；No Memory & FullText 在 reader 时构造。四组均是离线受控 replay，不直接等同各数据集官方 leaderboard。

| 数据集 / Reader | HAMGF | 最强记忆基线 | 配对差值 | 95% paired bootstrap CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| LoCoMo / Qwen | 56% | 62% HybridRAG | -6 pp | [-18,+6] | 0.5078 |
| LoCoMo / DeepSeek | 36% | 40% FullText | -4 pp | [-16,+8] | 0.7539 |
| LoCoMo / Gemini | 62% | 62% HybridRAG | 0 pp | [-10,+10] | 1.0000 |
| LongMemEval-S / Qwen | 52% | 50% HybridRAG | +2 pp | [-6,+10] | 1.0000 |
| LongMemEval-S / DeepSeek | 30% | 20% HybridRAG | +10 pp | [+2,+18] | 0.0625 |
| LongMemEval-S / Gemini | 50% | 54% HybridRAG | -4 pp | [-10,0] | 0.5000 |
| MA progressive / Qwen | 94% | 94% FullText | 0 pp | [0,0] | 1.0000 |
| MA progressive / DeepSeek | 78% | 72% HybridRAG | +6 pp | [-2,+14] | 0.3750 |
| MA progressive / Gemini | 82% | 76% FullText | +6 pp | [-2,+14] | 0.3750 |
| PrefEval / Qwen | 46% | 56% Mem0（MemOS 同分） | -10 pp | [-24,+4] | 0.2668 |
| PrefEval / DeepSeek | 36% | 64% HybridRAG | -28 pp | [-40,-16] | 0.000122 |
| PrefEval / Gemini | 56% | 80% MemOS | -24 pp | [-38,-10] | 0.00418 |

下述跨 reader 合并值用于描述趋势

| 数据集 | No Memory | FullText | HybridRAG | Mem0 | MemOS | HAMGF |
|---|---:|---:|---:|---:|---:|---:|
| LoCoMo | 9.33% | 51.33% | 51.33% | 41.33% | 41.33% | 51.33% |
| LongMemEval-S | 4.00% | 30.67% | 41.33% | 32.67% | 34.67% | 44.00% |
| MA progressive | 0.00% | 78.00% | 79.33% | 0.00% | 66.00% | 84.67% |
| PrefEval | 8.67% | 59.33% | 61.33% | 60.67% | 60.67% | 46.00% |

HAMGF 的平均 TTFT（ms）用于响应性需求统计：

| 数据集 | Qwen | DeepSeek | Gemini |
|---|---:|---:|---:|
| LoCoMo | 943 | 891 | 1605 |
| LongMemEval-S | 892 | 585 | 1395 |
| MA progressive | 1003 | 801 | 1435 |
| PrefEval | 780 | 890 | 2035 |

HAMGF 相对每个 reader 的最强记忆基线为 4 胜、2 平、6 负。四个正向差值均未通过未校正的 McNemar p<0.05；最强比较组是事后选择；LoCoMo 的 50 个问题来自十个会话。

## 公平性与可复现

- 仅使用过去会话；参考答案不进入 framework worker，也不进入生成提示词
- 当前 Mem0、MemOS抽取使用配置中的 **DeepSeek V4-Flash**，thinking 关闭、 temperature=0；主要负责Mem0的事实与关系抽取以及MemOS的结构化抽取和重组调度
- 常规抽取上限 8192 tokens，常规 reader/judge 为 192/384；PrefEval 为 384/512，特定 Judge 恢复为 1024。逐运行的最终配置以冻结 manifest 为准。
- HybridRAG、Mem0、MemOS 及 HAMGF v4 入口均使用 `text-embedding-3-small`； HAMGF v4 也采用 512-token、64-overlap 的 BM25+dense+RRF 入口，最多三个 seed，随后执行门控图扩展，不等同库内单入口 `ChainSearch`。
- 固定 k=6，所有检索组最多 3000 个 cl100k_base 证据 tokens，真实 API token usage 另记。
- HAMGF 的摘要/详情/边描述也计入同一预算。实际注入 ID、截断状态均保存。
- MemOS 的时间仅为合成 session-order 时间。
- 每 case 使用独立存储或随机 namespace。
- 主实验使用独立的 `v4-full-hamgf-event-graph` 目录。

## 其他

- “离线 replay”表示无实时任务环境，不表示无联网；真实模型与 embedding 仍通过 API 调用。
- 四主数据集检索子图不连通例数依次为 35/50、29/50、5/50、24/50；多入口输出可包含多个连通分量。
- MSC session 5、bundled shopping、group travel planning 在当前测试环境执行时存在一定问题因此未进入四组主排名。
