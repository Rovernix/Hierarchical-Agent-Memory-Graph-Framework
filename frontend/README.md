# HAMGF Studio frontend

此目录现在是 HAMGF Studio 的浏览器版本

## 本机启动

需要 Node.js 22.12+、Python 3.10+ 和原 HAMGF 的 `networkx` 依赖。在本目录执行：

```powershell
python -m pip install -e ..
npm ci
npm run build
npm start
```

`npm start` 使用构建生成的 `dist`，自动启动本地服务并打开浏览器。首次启动及修改界面后先执行 `npm run build`；仓库不提交构建产物。仅启动服务而不打开浏览器可用 `npm start -- --no-browser`。默认端口为 8765，可追加 `--port `

开发模式：

```powershell
npm run dev
```

首次使用在设置中填写 DeepSeek API Key。密钥保存在本地服务的数据目录；Windows 使用当前账户的 DPAPI 保护。聊天、逻辑记忆图谱和可选的 LLM 关系复核均使用新版 Studio 服务。
