# LawRAG 运行环境说明

> 文中阶段报告与完整错误档案属于历史过程材料，原件保留本地；当前成果与历史证据边界见 [RESULTS](RESULTS.md)，精选复盘见 [ENGINEERING_NOTES](ENGINEERING_NOTES.md)。

本文是复现与排错时使用的环境对照表，记录包管理器之外的运行事实。具体使用方法见项目根目录 [README](../README.md)，Python 包的精确版本以 [requirements.txt](../requirements.txt) 为准。

## 1. 所需软件

- Python 3.13；所有项目命令应显式使用项目虚拟环境的 `.\.venv\Scripts\python.exe`，不要依赖当前终端是否已经激活虚拟环境。
- Node.js 24.19.0 与 npm 11.17.0，用于 Vue 3 + Vite 开发与生产构建。
- Docker Desktop 29.8.0（WSL 2 后端），可选运行方式：不装 Python/Node 的机器可用 `docker compose build` + `docker compose up -d` 拉起完整应用，详见“Docker 运行方式”。
- 当前 PowerShell 执行策略禁止运行 `npm.ps1`，因此所有 npm/npx 命令使用 `npm.cmd` / `npx.cmd`；本次实际采用 `npm.cmd`，未修改执行策略。
- Windows 终端环境；当前项目在 PowerShell 下开发和验证。
- 性能采集与真实进程清理测试使用 Windows PowerShell 的 `Get-CimInstance Win32_Process` 查询父子进程关系，需要本机 CIM 查询权限；不再依赖 WMIC。查询失败或返回不完整时保持“未知”，不能据此宣布清理完成。
- Git 用于版本管理。Git 写操作是否可用还取决于当前执行环境，见“已知环境限制”。

## 2. 外部服务与凭据

凭据只从系统环境变量读取。这里只记录变量名，禁止在本文、截图、日志或命令输出中记录变量值。

| 服务 | 用途 | 当前模型 | 环境变量 |
| --- | --- | --- | --- |
| DeepSeek | 根据检索到的法条生成受资料约束的回答 | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| DeepSeek | 判断是否调用工具并生成工具参数 | `deepseek-flash` | `DEEPSEEK_API_KEY` |
| SiliconFlow | 查询与法条的向量化 | `BAAI/bge-m3` | `SILICONFLOW_API_KEY` |
| SiliconFlow | 对向量召回候选进行重排 | `BAAI/bge-reranker-v2-m3` | `SILICONFLOW_API_KEY` |
| 博查 Web Search | 法规时效核验与通用联网搜索 | Web Search API | `BOCHA_API_KEY` |
| Langfuse（可选） | LLM 调用链路可观测性追踪 | 可观测性平台，非模型 | `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST` |

环境变量存在性一律通过 `check_env.py` 验证；该脚本只报告变量名和读取状态。

### MCP 子进程环境（2026-09-20 实测）

MCP stdio 子进程（`app/mcp_bocha_server.py`）的最终环境由两层组成：

```text
MCP 子进程最终环境 = MCP SDK 安全默认环境 + LawRAG 应用显式环境
```

- **MCP SDK 自动继承（mcp==1.30.0 的 `DEFAULT_INHERITED_ENV_VARS`，共 12 项）**：`APPDATA`、`HOMEDRIVE`、`HOMEPATH`、`LOCALAPPDATA`、`PATH`、`PATHEXT`、`PROCESSOR_ARCHITECTURE`、`SYSTEMDRIVE`、`SYSTEMROOT`、`TEMP`、`USERNAME`、`USERPROFILE`。由 SDK 自行选择，本应用不修改该常量。
- **LawRAG 应用显式透传（正向白名单）**：
  - 代理变量：`HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`、`NO_PROXY` 及其小写形式（仅复制父环境中实际存在的项）；
  - SSL 证书变量：`SSL_CERT_FILE`、`SSL_CERT_DIR`（仅复制实际存在的项）；
  - `BOCHA_API_KEY`（从父环境读取，缺失时为空串；用户配置方式不变）；
  - 固定 UTF-8 配置：`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`（父环境不能覆盖）。

- 应用不再复制完整父环境；`DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY`、OpenAI/AWS/GitHub 令牌、`PYTHONPATH` 等无关变量一律不传给博查 MCP 子进程。
- 代理变量值可能包含敏感信息，本文件、测试与报告均不记录其值。

### Langfuse 可观测性（2026-09-24 实测）

- **用途**：把全部真实 LLM 调用的完整链路（提示词、回答、token 用量、延迟）上报到 Langfuse 平台。覆盖五类调用：追问改写（`rewrite`）、工具路由（`router`）、HyDE 假想文本（`hyde`）、多查询改写（`multi-query`）、最终答案生成（`generate`）。
- **接入方式**：langfuse.com 注册（免费档即可）→ 创建项目 → 一对密钥只用 `setx` 写入用户级环境变量，`LANGFUSE_HOST` 必须与账号区域一致（美区为 `https://us.cloud.langfuse.com`），设置后新开终端。私钥只在创建时完整显示一次。
- **fail-open 语义**：`LANGFUSE_PUBLIC_KEY` 或 `LANGFUSE_SECRET_KEY` 任一缺失即自动关闭可观测性，问答主流程完全不受影响；handler 构造失败也静默降级；回调内部异常由 LangChain 回调管理器吞掉（`raise_error` 默认 False）。
- **实现**：`app/observability.py` 提供 `traced_invoke()` 统一包装；会话与标签经 LangChain 调用 metadata 传递（`langfuse_session_id` / `langfuse_tags`，langfuse v4 约定）；同一会话的多次调用在 Langfuse 控制台 Sessions 视图分组呈现。显式调用模型时经 `ensure_config()` 合并继承配置，对继承回调**追加而非替换** Langfuse handler（E-075），保证 LangGraph `stream_mode="messages"` 的 token 不被 Langfuse 回调覆盖；初始化异常日志经 `_redact_secrets()` 脱敏后输出异常摘要，不再输出未脱敏堆栈（E-076）。
- **测试隔离**：`tests/conftest.py` 以 autouse 夹具自动禁用 Langfuse，防止宿主机真实密钥导致测试发送真实 trace（E-074 纪律）；`tests/test_observability.py` 全部桩化，不联网。日志脱敏 `_redact_secrets` 已覆盖 `_SECRET_KEY` 后缀。
- Langfuse 密钥不参与 `check_env.py` 的必需项校验；未配置属正常"可观测性关闭"状态，不是环境错误。

## 3. 数据前提

- `kb/` 是原始法律语料目录，当前包含两份 DOCX。
- `chroma_db/` 是执行入库后生成的本地 Chroma 持久化目录，不进入 Git。
- BM25 使用 jieba 和 rank-bm25，从当前 Chroma 集合的同一批节点构建进程内索引；Web 应用启动时先加载向量索引，再预热并缓存 BM25 索引，不生成新的持久化目录，也不调用外部服务。
- 首次使用、语料新增、语料替换或语料正文更新后，必须重新运行入库命令。
- Web 服务在启动时加载并缓存向量索引和 BM25 索引；重新入库后应重启 Web 服务，使进程使用新索引。

## 4. 实测环境

以下结果核心项于 **2026-09-16** 实测；Langfuse 可观测性相关新增项（`langchain`、`langfuse` 依赖与行为）于 **2026-09-24** 实测。

| 字段 | 实测值 |
| --- | --- |
| Python | CPython 3.13.3，解释器为项目 `.venv` 中的 `python.exe` |
| Node.js | v24.19.0 |
| npm | 11.17.0；PowerShell 中通过 `npm.cmd` 调用 |
| 操作系统 | Windows 10，版本 `10.0.19045`，Service Pack 0 |
| 终端 | PowerShell 7.6.5 |
| Docker | Docker Desktop 29.8.0（Client/Server 29.8.0，WSL 2 后端）；`lawrag-lawrag` 镜像 1.26GB（2026-09-24 实测） |
| Python 依赖 | `langchain` 1.4.2、`langchain-core` 1.6.3、`langchain-mcp-adapters` 0.3.2、`langchain-openai` 1.6.2、`langfuse` 4.15.4、`langgraph` 1.2.11、`langgraph-checkpoint-sqlite` 3.1.1、`mcp` 1.30.0；其余以 [requirements.txt](../requirements.txt) 的 `==` 锁定版本为准 |
| 未声明的传递依赖 | `websockets` 16.1.1；由 LangGraph 依赖树解析得到，不在 `requirements.txt` 中直接声明 |
| 向量化模型 | `BAAI/bge-m3` |
| 重排模型 | `BAAI/bge-reranker-v2-m3` |
| 生成模型 | `deepseek-chat` |
| 向量库 | Chroma 1.5.9；本地目录 `chroma_db/`，集合名 `lawrag` |
| 当前索引 | 由当前 `kb/` 两份 DOCX 入库生成；集合实测为 161 个法条节点，距离空间为 cosine |
| 关键词索引 | jieba 0.42.1 与 rank-bm25 0.2.2；Web 启动时在向量索引之后从同一批 161 个 Chroma 节点构建，进程内缓存一次 |
| 默认检索策略 | `hybrid_rerank`：向量与 BM25 各召回候选，经 RRF 融合后重排取 top3 |
| 多轮状态 | LangGraph 五节点条件编排；SQLite 会话元数据与 Checkpointer 消息历史保存在 `data/lawrag.sqlite3` |
| 本次验证日期 | 2026-10-10（核心项 2026-09-16 实测保持不变；评测相关新增项 2026-10-09 实测；性能基线测量运行方式 2026-10-10 实测，R1、R2 返工后复测） |

### 可配置的模型环境变量与外部工具

- `LAWRAG_CHAT_MODEL` 控制最终回答模型，默认 `deepseek-chat`；`LAWRAG_ROUTER_MODEL` 控制工具路由模型，默认 `deepseek-flash`。可通过 `setx LAWRAG_CHAT_MODEL "模型名"` 与 `setx LAWRAG_ROUTER_MODEL "模型名"` 设置，执行 `setx` 后必须新开终端，已运行的进程不会自动获得新值。
- 两者取值为空串时按“未配置”处理并回退默认模型名（Docker Compose 的 `${VAR:-}` 会把未设置的变量传成空串，见 E-083（未公开的历史记录））。
- DeepSeek 官方 `GET /models` 当前未列出 `deepseek-chat`，但该遗留别名仍可调用。长期建议迁移到官方当前列出的 `deepseek-flash` 或 `deepseek-v4-pro`，迁移前应单独验证回答质量与成本。
- `BOCHA_API_KEY` 供法规时效核验与联网搜索使用，可执行 `setx BOCHA_API_KEY "你的密钥"` 后新开终端。额度不足时博查会返回 HTTP 403；薄 MCP server 会把它转换为可读文本，问答接口仍正常返回而不是抛出 500。
- 本项目不使用 `.env` 保存这些值；凭据只从进程环境读取，不应写入仓库、日志或报告。

本轮流式输出与会话侧边栏没有新增 Python 或 npm 依赖，继续使用当前锁定环境。以下行为已在当前真实库版本中实测：

1. `graph.stream(..., stream_mode="messages")` 产出 `(AIMessageChunk, metadata)`，节点名位于 `metadata["langgraph_node"]`。
2. 节点内普通 `.invoke()` 会自动进入流式路径，不需要改成 `.stream()`，也不需要给模型设置 `streaming=True`。
3. 同时按 generate 节点与 `AIMessageChunk` 类型过滤后，token 顺序拼接逐字等于最终 `state["answer"]`。
4. 流末尾可能出现 `content` 为空字符串的 chunk，必须跳过。
5. `prepare` 节点返回的消息也会发射，必须按 `langgraph_node` 过滤，只转发 `generate` 节点。
6. `stream()` 完成后立即 `get_state()` 可以可靠读取 citations，不需要增加 durability 参数。
7. `SqliteSaver.delete_thread(thread_id)` 在尚未创建表时调用不会抛异常，不需要额外捕获。
8. `StreamingResponse` 支持同步生成器，Starlette 会在线程池中迭代，因此端点可保持同步 `def`。
9. generate 节点除了模型产生的多个 `AIMessageChunk`，还会发射节点返回的新建完整 `AIMessage`；新 id 绕过按 id 去重，因此节点名过滤之外还必须做消息类型过滤。
10. Checkpointer 回放 citations 时，外层可保持 `NodeWithScore`，但 `.node` 会退化为 dict，`.score` 仍为 float；呈现层必须兼容活 `TextNode` 与字典节点两种形态。

### 当前语料版本

| 法律 | 仓库文件 | 版本日期 | 来源 |
| --- | --- | --- | --- |
| 《中华人民共和国劳动合同法》 | `kb/中华人民共和国劳动合同法.docx` | 2012-12-28 修正；DOCX 正文同时记载 2007-06-29 通过 | 法律法规数据库；仓库未保存具体下载 URL |
| 《中华人民共和国消费者权益保护法》 | `kb/中华人民共和国消费者权益保护法.docx` | 2013-10-25 第二次修正；DOCX 正文同时记载 1993-10-31 通过、2009-08-27 第一次修正 | 法律法规数据库；仓库未保存具体下载 URL |

“法律法规数据库”来自仓库现有设计记录；法律版本日期来自当前 DOCX 正文。由于文件和仓库没有保留更具体的来源 URL，本文不推测下载地址。

## 5. 启动步骤

以下命令均在项目根目录执行。Python 命令始终使用项目 `.venv` 的显式解释器路径。

### 1. 创建虚拟环境（首次使用）

先安装 Python 3.13，再创建项目环境：

```powershell
py -3.13 -m venv .venv
```

后续使用显式解释器即可；终端策略允许时，也可选择激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

当前实测环境会因 PowerShell 执行策略阻止该脚本。激活不是后续步骤的必要条件，继续使用下面的显式解释器命令即可，不需要修改系统执行策略。

### 2. 安装锁定依赖

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. 校验环境变量

```powershell
.\.venv\Scripts\python.exe -X utf8 .\check_env.py
```

脚本应只显示 `DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY` 及其“已读取”状态，不应显示任何值。

### 4. 首次使用或语料更新后入库

```powershell
.\.venv\Scripts\python.exe -m app.ingest
```

已有与当前语料一致的 `chroma_db/` 时可跳过本步。

### 5. 构建 Vue 生产版

```powershell
npm.cmd --prefix frontend install
npm.cmd --prefix frontend run build
```

`frontend/dist/` 是被 Git 忽略的本地构建产物。单端口运行必须先完成本步，再启动后端。

### 6. 启动后端 Web 服务（单端口）

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

保持服务终端运行，然后访问：

- Vue 生产版入口：<http://127.0.0.1:8000>
- 极简演示页：<http://127.0.0.1:8000/minimal>
- FastAPI 接口文档：<http://127.0.0.1:8000/docs>

不要通过 `file://` 直接打开 `app/static/index.html`；该方式没有可连接的 `/api/ask` HTTP 服务。

未执行构建时，`GET /`、`/assets/*` 与 `/favicon.ico` 返回 404，但 FastAPI 仍可启动，`/api`、`/minimal` 和 `/docs` 不受影响。静态挂载是否注册在后端进程启动时决定；如果先启动后端再构建前端，必须重启后端。`--reload` 默认监听 Python 源文件，不会因为 `frontend/dist/` 出现而自动重新加载应用。

### 7. 启动 Vue 开发服务器

另开一个项目根目录下的 PowerShell 终端：

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

访问 <http://localhost:5173>。Vite 将 `/api` 请求代理到 <http://127.0.0.1:8000>，因此后端必须同时运行。生产模式则由 FastAPI 在 `/` 直接托管 `frontend/dist/`，前端与 API 同源。

### 8. 运行后端测试（推荐入口）

推荐用项目自带的测试入口运行；它在每次调用时于 `logs/`（已被 Git 忽略）下各自分配一个独立的新临时目录作为 `--basetemp`，并默认关闭 pytest cacheprovider，从而避开对固定临时目录/缓存路径的依赖。入口只启动一次 pytest 并把其退出码原样传回，不做自动重试。

```powershell
# 本地全套测试（正式测试集为 tests/）
.\.venv\Scripts\python.exe scripts\run_tests.py tests -q
# 只跑单个测试文件：
.\.venv\Scripts\python.exe scripts\run_tests.py tests/test_tools.py -q
```

入口自行管理 `--basetemp`，传入冲突参数会被明确拒绝且不会启动 pytest。收集范围说明：正式测试集为 `tests/`；根目录 `pytest.ini` 已设置 `testpaths = tests`，因此无路径参数的 `python -m pytest`（本地与 CI 同一命令）也只收集 `tests/`，不会遍历 `logs/`（2026-09-24 起，参见 E-054（未公开的历史记录））；`logs/` 下若存在临时验收/调试探针，仍应按明确路径单独执行，不要使用 `test_` 前缀命名（参见 E-052（未公开的历史记录））。CI 说明：GitHub Actions（windows-latest + Python 3.13）在全新检出环境运行 `python -m pytest`——没有 `.venv`、`data/`、`chroma_db/` 与任何密钥环境变量，测试不得依赖这些仅存在于开发机的状态（参见 E-074（未公开的历史记录））。另外 `tests/conftest.py` 的 autouse 夹具会自动禁用 Langfuse，宿主机存在真实密钥时测试也不会发送真实 trace。实测结果（2026-09-24 CI 模拟 282 通过；2026-10-09 新增评测测试后本地全套 313 通过）。相关背景见 E-044（未公开的历史记录）。

### 9. Docker 运行方式（2026-09-24 实测）

本节是第 1—6 步的容器替代方案：不装 Python/Node 的机器只需 Docker Desktop，即可用两条命令拉起完整应用（前端在镜像内构建）。容器内应用与本地运行共用同一份代码与同一套环境变量名。

```powershell
# 构建镜像（多阶段：node:24-slim 构建前端 → python:3.13-slim 运行）
docker compose build
# 启动（后台运行 + 健康检查 + 自动重启）
docker compose up -d
# 宿主机没有 chroma_db/ 时，首次在容器内入库（写入挂载目录）
docker compose run --rm lawrag python -m app.ingest
```

- 访问地址与本地一致：<http://127.0.0.1:8000>（容器内 uvicorn 绑定 `0.0.0.0:8000`，端口映射 `8000:8000`）。
- 健康检查：`curl http://127.0.0.1:8000/healthz` 返回 `{"status":"ok"}`；`docker compose ps` 的 STATUS 列显示 `healthy`。该端点不依赖索引、数据库与外部服务，容器刚启动、索引尚未加载时也可响应（索引未就绪时业务端点仍按原行为返回 503）。
- 密钥来源：宿主机进程环境变量，由 Compose 读取后注入容器，不写入镜像与文件（本项目不使用 `.env`）。启动终端必须是“能看到这些变量”的进程：Compose 只读取启动它的那个进程的环境（E-082（未公开的历史记录） 同类问题），早于 `setx` 启动的终端或 agent 进程即使“新开窗口”也可能拿不到新值，必要时用 `$env:DEEPSEEK_API_KEY = ...` 在当前进程显式注入后再 `docker compose up -d`。
- 数据持久化：`chroma_db/` 与 `data/` 以宿主机相对路径绑定挂载到容器同名路径，容器重建（`docker compose down && docker compose up -d`）后会话历史与向量索引仍在。
- 实测结果（2026-09-24）：`docker compose build` 成功（无需额外编译工具，纯 wheel 安装）；`docker compose ps` 显示 `healthy`；`/healthz` 返回 200 与 `{"status":"ok"}`；浏览器打开 <http://127.0.0.1:8000> 正常渲染（标题、输入框、免责声明、侧边栏会话列表齐全，控制台无报错）；一次真实流式问答产出 162 个 token、唯一 done、3 条引用，且该会话的 trace（GENERATION）出现在 Langfuse 控制台。

已知限制：

- 绑定挂载要求宿主机存在 `chroma_db/` 与 `data/` 两个目录：`data/` 由容器首次启动（或首次问答）自动创建，`chroma_db/` 由容器内入库命令写入后出现在宿主机；两个目录都不入 Git，全新克隆的机器需先入库。
- 镜像体积 1.26GB（`chromadb`、`onnxruntime`、`tokenizers` 等 Python 依赖占主要部分），首次构建需下载大量依赖；`python:3.13-slim` 对本项目依赖足够，实测无需安装 `gcc`。
- 容器内 `logs/` 为临时目录（未挂载），容器重建后错误日志不保留；需要留存时用 `docker compose logs` 导出。

### 10. 运行回答质量与工具路由评测（2026-10-09 实测）

评测 CLI 为 `tests/run_quality_evaluation.py`，题集为 `tests/evaluation_cases.json`（40 场景，版本 1.0.0），
详见 [EVALUATION.md](EVALUATION.md)。不新增 Python 依赖，仅用标准库 + 现有依赖 + pytest。

```powershell
# 题集离线校验（不访问网络，输出题集与评分脚本的冻结 SHA-256）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py validate
# 单场景真实采集 / 全集真实采集（需要 DEEPSEEK_API_KEY、SILICONFLOW_API_KEY）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py collect --run-id eval-yyyymmdd --case-id TOOL-CALC-01
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py collect --run-id eval-yyyymmdd
# 离线汇总（读取已保存结果与评分记录生成 report.md，不调用模型）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py report --run-id eval-yyyymmdd
# R1 起：历史运行的修正规则重算（显式转换到新目录，原目录只读；核验来源并记录 provenance）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py rescore --source-run <旧run> --run-id <新run> --reason "..."
```

隔离方式（评测进程内完成，不改用户系统环境、不加应用配置项）：`LAWRAG_DB_PATH` 指向
`logs/evaluation/<run_id>/lawrag.sqlite3` 并改绑 `app.db.DATABASE_PATH`、清空连接缓存；
Chroma 使用 `logs/evaluation/<run_id>/chroma_copy` 隔离副本（复制前后比对源快照检测并发入库，
不一致即中止），改绑 `app.ask.CHROMA_DIR` 并清空索引缓存；Langfuse 上报默认关闭；
采集前后对真实 `chroma_db/`、`data/`、`kb/` 做只读快照比对并写入 manifest。
运行产物在 `logs/evaluation/<run_id>/`（manifest.json、results.jsonl、reviews.json、report.md），不入 Git。
已知限制：真实采集依赖外部模型/重排服务可用；缺凭据时保留已完成部分并标记未完成原因。
R1 返工（2026-10-09）后：每次 attempt 使用独立会话；报告只读运行目录冻结题集快照（cases_frozen.json）；
复核记录绑定结果内容 hash，报告不修改复核记录；详见 [EVALUATION.md](EVALUATION.md) §8。

### 11. 运行请求性能与调用成本基线测量（2026-10-10 实测）

第三阶段第一步的测量 CLI 为 `tests/run_performance_baseline.py`，固定场景为
`tests/performance_cases.json`（4 场景 6 轮，与评测题集独立），详见 [PERFORMANCE.md](PERFORMANCE.md)。
不新增 Python 依赖，仅用标准库 + 现有依赖 + uvicorn。

```powershell
# 离线校验场景并输出冻结哈希（不访问网络）
.\.venv\Scripts\python.exe tests\run_performance_baseline.py validate
# 真实采集：启动本次专用隔离服务进程，真实 HTTP/SSE 执行预热 1 遍 + 正式 3 遍
.\.venv\Scripts\python.exe tests\run_performance_baseline.py collect --run-id perf-yyyymmdd
# 离线汇总：只读运行目录已保存数据，派生结果写入**新的输出目录**（R1 起绝不回写运行目录原始文件）
.\.venv\Scripts\python.exe tests\run_performance_baseline.py report --run-id perf-yyyymmdd --output-dir <新目录>
```

- 服务进程由 `serve` 子命令在隔离数据上拉起（`LAWRAG_DB_PATH` 指向运行目录、Chroma 用运行目录内副本并改绑 `app.ask.CHROMA_DIR`），使用本次分配的空闲本地端口，不占用或停止用户已有服务；测量结束只终止本次进程，启动失败（超时/提前退出/异常）也清理本次进程并留存清理证据。
- 服务端观测由应用内记录器按 `LAWRAG_PERF_OBS_FILE` 门控写入运行目录 `observations.jsonl`；该变量未设置时为空操作，不影响问答主流程（默认关闭）。
- 测量期间 Langfuse 统一关闭（子进程环境显式清空两项密钥），本地计量保留。
- 运行产物在 `logs/performance/<run_id>/`（manifest.json 含采集前后真实目录逐文件快照与清理证据、requests.jsonl 含原始 SSE 时间线、observations.jsonl、report.md、chroma_copy、lawrag.sqlite3、server.log），不入 Git；新执行使用新目录，不覆盖旧结果。
- R1 起 `report` 的派生输出（report.md、requests_derived.jsonl）写入 `--output-dir` 指定的新目录（默认 `logs/performance/<run_id>-report-<时间戳>`），同输入重复重建输出字节一致。
- R2 起 report 输出结构（文件不变，表格口径调整）：任务表增加计划/实际轮次与状态列（完整成功 / 已执行未完成或失败 / 未执行），逐轮表增加"未执行"列、无成功样本时统计列为 N/A，费用表逐调用计费并在部分已知时标注"已知部分"；计划任务/计划请求来自 manifest 冻结计划（含未执行任务行）。R3 起按槽位身份统计：整体/逐轮/场景的计划列只来自冻结槽位，未执行=空槽位数，计划外/重复记录在"计划外/重复异常记录"表单列（含身份），不进计划口径。2026-10-10 实测（report CLI 缺单项费率可正常出报告、重复重建字节一致；计划外记录用例经 report CLI 验证）。
- 已知限制：真实采集依赖外部模型/重排/搜索服务可用；每次场景重复用新会话、多轮场景内共享会话；3 次重复只够初步基线，不发布 P95，不开展并发压测。

## 6. 已知环境限制

- 本机 Git for Windows 版本为 2.55.0，但配套命令环境不完整，实测缺少 `file` 等 coreutils。对 DOCX 执行依赖 textconv 的历史 diff 可能失败，详见 E-003（未公开的历史记录）。
- Codex 桌面端沙箱下 `.git` 为只读；`git add`、`git commit`、`git push` 等 Git 写操作由用户手动执行。
- 项目依赖必须安装到项目 `.venv`，不得直接安装到共享 Python 环境，详见 E-010（未公开的历史记录）。
- 当前 PowerShell 执行策略会阻止 `.venv\Scripts\Activate.ps1`；使用 `.\.venv\Scripts\python.exe` 显式路径即可，无需修改系统策略，详见 E-014（未公开的历史记录）。
- 同一执行策略也会阻止 `npm.ps1`；前端命令应使用 `npm.cmd`，无需修改系统策略。
- Vue 静态资源挂载是否注册在后端进程启动时决定；新增或恢复 `frontend/dist/` 后必须重启后端，不能依赖 `--reload` 感知构建产物变化。

## 2026-10-11 授权真实基线补采入口

本次 24 次业务请求（预热 6、正式 18；联网场景执行 4 次）使用专用包装器：
` .\.venv\Scripts\python.exe outputs\live_baseline_20261011\run_capture.py --run-id perf-20261011-live1 `。
包装器复用已验收的采集、隔离、清理及报告实现；首个失败或缺 session 的请求落盘后即停止，不补跑。供应商内部既有重试不变，24 次业务请求不是供应商调用硬上限。采集器与包装器/价格文件哈希均写入 manifest。
价格冻结于 `outputs/live_baseline_20261011/pricing.json`：只核实当前页面支持的模型费率；其余 UNKNOWN。Langfuse 仍关闭，不新增依赖，不改业务模型、提示词或检索策略。离线检查与真实执行结果见本次交付报告，不把价格目录估算称为实际扣费。验证日期：2026-10-11。

## 2026-10-11 OPT1 优化后真实对照复测入口（历史准备记录）

以下保留采集前的准备约定；同日已完成的真实复测及其验收结论见 [RESULTS](RESULTS.md#4-性能缺参追问跳过无效检索)。后续重新采集仍需单独授权。

优化后对照采集使用专用薄包装器（不覆盖旧包装器、不扩建通用采集框架）：
` .\.venv\Scripts\python.exe outputs\live_opt1_20261011\run_capture_opt1.py --run-id perf-YYYYMMDD-opt1-1 `。
包装器复用已验收采集器（tests/run_performance_baseline.py，同一模块实例）与已验收首错停止逻辑（outputs/live_baseline_20261011/run_capture.py::guarded_samples）；**实际 collect 读取 `tests/performance_cases_opt1.json`**（CASES_PATH 显式指向该文件，manifest 的 cases_sha256/cases_file_sha256 与场景身份来自该文件），并把实际场景文件、包装器、价格文件与复用来源哈希写入 manifest.code_hashes。
固定计划与优化前一致：4 场景 6 轮，预热 1 遍 + 正式 3 遍 = 24 业务请求（预热 6、正式 18）/ 16 任务（预热 4、正式 12）；联网场景执行 4 次。首条预热同时作为冒烟，不额外增加请求；首个请求失败或缺 session 落盘后即停止（stop.json），不重试整轮。价格依据冻结于 `outputs/live_opt1_20261011/pricing.json`（2026-10-11 快照，含未核实项；若实际执行日不同需复核费率）。输出目录 `logs/performance/<run_id>/`（新 run_id，采集器拒绝覆盖非空旧目录，不触碰 perf-20261011-live1 与 r1/r2 证据）。离线入口验证见 tests/test_opt1_post_capture_20261011.py 与 outputs/STAGE3_OPT1_POST_PREPARATION_20261011.md。验证日期：2026-10-11。

## 展示整理验证（2026-10-12）

README 增补首次创建虚拟环境的步骤；没有改变模型、语料、Python 依赖或服务启动机制。完整错误档案改为本地文件，归档范围检查改用 `--baseline` 显式快照，见 [工程复盘](ENGINEERING_NOTES.md#本地完整档案)。本轮离线验证结果见本地交付记录；表中历史环境与模型实测日期保留，不当成本轮重新测量。

本轮实测 Python 3.13.3、Node.js 24.19.0。从暂存区导出代码，在无本地索引、私人日志及 API 凭据的目录中运行：Python 回归 567 通过、1 跳过、2 条依赖弃用警告。跳过项为真实历史记录身份校验，已在保留历史数据的原工作区单独通过；前端逻辑测试 9 通过，冻结题集 40 场景校验通过。修复了测试自身对本机目录和模型凭据的依赖；没有调用真实模型，也未执行新的入库、完整服务启动或 Docker 验证。Windows 真实进程清理用例使用可读取 WMIC 进程表的正常权限环境。

## Windows 进程查询兼容性验证（2026-10-13）

GitHub Actions 的 Windows Server 2025 环境未能通过原 WMIC 进程查询测试。进程枚举现改为系统自带的 `powershell.exe -NoProfile -NonInteractive` 调用 CIM，并严格检查 JSON 中的子、父进程编号；30 秒查询超时与清理失败判定保持不变，不安装额外 Python 包或 Windows 可选组件。

本机 Python 3.13.3 的性能回归 98 项通过，完整离线回归 578 项通过（2 条依赖弃用警告），包含真实父子进程退出与端口释放、查询超时、命令失败、无效或不完整进程表等检查。该结果只验证离线测试和清理行为，不代表重新完成真实模型性能测量；历史测量值及证据不变。

## 公开快照语料元数据核验（2026-10-19）

公开副本清除了两份 DOCX 的作者、最后修改者、自定义属性及公司等文档属性。逐个 ZIP 成员核对确认仅 docProps 下的属性 XML 发生变化，word/document.xml 与其余正文资源逐字节相同；法律名称、版本日期与来源沿用上表，未改写法律正文。

本次未运行付费入库、模型调用或 Docker，未重新测量历史服务版本与性能。公开仓库不携带索引；首次运行仍需按现有步骤入库。私人原始语料和索引未修改。

公开副本离线实测：Python 测试 580 项通过、1 项跳过（历史数据身份校验，公开副本不含私人历史记录）、2 条依赖弃用警告；前端逻辑测试 9 项通过，冻结题集 40 个场景校验通过。验证日期：2026-10-19。
