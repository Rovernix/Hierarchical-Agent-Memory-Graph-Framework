# HAMGF CMG Visualizer

React + Cytoscape 的 Chain-of-Memory Graph 可视化 Demo 与可嵌入插件。

从项目根目录安装 Python 包并在另一个终端启动 API：

~~~bash
python -m pip install -e .
hamgf-api --demo
~~~

以下命令在 frontend/ 目录执行。Vite 不会自动启动 Python API。

```bash
npm ci
npm run dev
```

默认连接 `http://127.0.0.1:8000`；可通过`VITE_HAMGF_API_URL` 指定地址。API 不可用时页面使用标准 CMG 演示图，并明确标记为“演示数据”。

```bash
npm test       # 图转换、分叉/回溯、修正链逻辑
npm run build  # 同时构建 dist/ Demo 与 dist-plugin/ 插件
```

插件源码入口是 `src/plugin/index.js`，导出 `CmgMemoryPlugin`、REST 客户端和无 UI 的图转换工具。

请不要把长期凭据写进 Vite 环境变量并发布给浏览器（ye）。完整接入见[可视化文档](../docs/visualization-and-integration.md)。
