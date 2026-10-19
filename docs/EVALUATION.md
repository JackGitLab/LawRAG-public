# LawRAG 回答质量与工具路由评测

> 文中阶段报告与完整错误档案属于历史过程材料，原件保留本地；当前成果及固定历史链接见 [RESULTS](RESULTS.md)，精选复盘见 [ENGINEERING_NOTES](ENGINEERING_NOTES.md)。

本文档说明新增冻结评测集（40 个场景）的构成、标注来源、运行命令、指标口径、数据隔离方式与限制。
评测实现位于 `tests/run_quality_evaluation.py`，题集位于 `tests/evaluation_cases.json`，
针对性测试位于 `tests/test_evaluation_cases.py` 与 `tests/test_quality_evaluation.py`。

本阶段验收的是**评测实现与证据真实性**，不要求模型成绩达到某个预设百分比。

## 1. 评测目标

完成后应能回答：

1. 新问题能否检索到必要法条？（Hit@3 + 逐题命中/缺失法条）
2. 生成结论是否得到证据支持，引用是否正确？（引用支持率，人工复核）
3. 资料不足时是否恰当拒答，是否出现边拒答边编造？（拒答与边界处理，人工复核）
4. 工具选择和参数抽取是否正确？（工具选择正确率、参数正确率）
5. 多轮场景是否完成预先定义的任务？（场景完成率，按冻结的 completion_criteria 复核）

## 2. 题集构成（tests/evaluation_cases.json，版本 1.0.0）

| 类别 | 数量 | 内容 | 场景 ID |
| --- | --- | --- | --- |
| 库内问答 library_qa | 16 | 《劳动合同法》8 题、《消费者权益保护法》8 题；覆盖口语表达、条件与例外、多条文组合 | QA-LAB-01..08、QA-CON-01..08 |
| 回答边界 boundary | 8 | 4 个库外问题（BND-OOS）、4 个信息/证据不足场景（BND-INS） | BND-OOS-01..04、BND-INS-01..04 |
| 工具路由 tool_routing | 8 | 4 个完整计算参数（TOOL-CALC）、2 个缺失必要计算参数（TOOL-MISS）、2 个联网路由（TOOL-WEB，route_only） | TOOL-CALC-01..04、TOOL-MISS-01..02、TOOL-WEB-01..02 |
| 多轮对话 multi_turn | 8 | 每场景固定两轮：指代（MT-REF）、省略（MT-ELL）、事实纠正（MT-COR）、主题切换（MT-SWT）各 2 个 | MT-REF/ELL/COR/SWT-01..02 |

- 问答图（graph）场景 38 个，其中 8 个两轮，共 46 次问答图执行；route_only 场景 2 个。
- 每个场景字段：`id`、`category`、`subcategory`、`execution_scope`、`annotation_status`、`turns`
  （每轮 `question`、`use_web_search`、`expected_articles`、`required_points`、`forbidden_claims`、
  `expected_tools`，缺参轮另有 `missing_params`）、`completion_criteria`、`evidence`、
  `difference_from_regression`（与旧 55 题的差异说明）。

### 2.1 与原有 55 题回归集的关系

- 新题独立于 `tests/regression_questions.json`，逐题记录差异（组合条文、例外条款、边界条件、
  多轮动态等），离线校验会拒绝与旧题的明显重复（归一化后相等或包含）。
- 「独立」仅指独立于现有回归集、且未用于本轮调优；**不是**经过外部专家审核的盲测集。

### 2.2 标注来源与可信度

- 出题与标注由执行 agent 基于 `kb/` 两部法律原文完成；每题 `evidence` 记录来源、条号与
  支持标注的短摘录，离线校验逐条比对摘录必须是 kb/ 原文子串，且 `expected_articles`
  指向的条文必须真实存在于 kb/。
- `annotation_status` 两类：
  - `executor_annotated`：可直接由条文原文支持；
  - `needs_legal_review`：条文适用存在解释空间（QA-LAB-03、QA-LAB-08、QA-CON-05、TOOL-CALC-04），
    已附 `annotation_note`，**不作可信金标准**，待独立复核。
- 缺参场景（TOOL-MISS）只评估「不编造参数、不执行缺乏依据的计算」，不要求尚未实现的
  结构化追问能力。
- 预期答案、法条与评分规则只供评分使用：采集时仅向被测应用传入 `question`、
  `use_web_search`、`session_id`（测试 `test_expected_fields_never_enter_model_input` 固化）。

### 2.3 冻结记录（首次真实采集前，2026-10-09）

| 冻结对象 | SHA-256 |
| --- | --- |
| tests/evaluation_cases.json | `ebefe69fb2d80d5794c1df540bcf895d13c04638114718c65b0644f66c330856` |
| tests/run_quality_evaluation.py（含评分规则） | `138c732d1aa6272a88081168e709f62a93e44438fc9d3a9e3b72ab59dcc4f55f` |

- 采集时 `manifest.json` 会重新记录两个哈希；若评测后发现标注错误，建立新版本题集并说明原因，
  保留旧结果，不删题、不放宽标注、不挑选最佳运行结果。
- R1 返工后的评分脚本 SHA-256 见 `logs/evaluation/eval-20261009-r1/conversion.json`
  （修正规则重算使用，不冒充原始采集时的评分脚本 hash）。

### 2.4 身份字段说明

| 字段 | 含义 | 与谁比较 |
| --- | --- | --- |
| `cases_frozen.json` 文件字节 SHA-256（= 规范 JSON 内容 hash，如 `1eccac63…`） | 运行目录内的冻结题集身份（快照按 `json.dumps(indent=2, sort_keys=True)` 字节写入，故字节 hash 与内容 hash 相等） | `manifest.code_sha256['tests/evaluation_cases.json']`、`conversion.frozen_cases_sha256`、`reviews.cases_sha256` |
| 工作区 `tests/evaluation_cases.json` 原始文件字节 SHA-256（如 `ebefe69f…`） | 采集时的原始题集文件身份（v1 manifest 记录值） | 仅在 `rescore` 来源核验时与工作区原始文件/内容比对；**不与冻结 hash 判等**（二者是不同字段） |
| `result_hash` | 单条结果的内容身份（`RESULT_HASH_FIELDS` 规范 JSON 的 SHA-256） | 保存 hash ↔ 按当前内容重算 hash ↔ `reviews.result_hash`/`result_hashes` 三方必须一致 |

## 3. 运行命令（项目解释器，不激活全局环境）

```powershell
# 1) 题集离线校验（不访问网络；同时输出冻结哈希）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py validate

# 2) 单场景真实采集（示例：一道完整参数计算题）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py collect --run-id eval-20261009 --case-id TOOL-CALC-01

# 3) 全集真实采集（38 个问答图场景 + 2 个 route_only 场景）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py collect --run-id eval-20261009

# 4) 定向补齐未完成项（追加为新的 attempt，不覆盖旧记录）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py collect --run-id eval-20261009 --case-id BND-OOS-02,MT-COR-01

# 5) 离线汇总（读取已有结果与评分记录，生成 report.md，不调用模型）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py report --run-id eval-20261009

# 5b) R1 起：历史运行的修正规则重算（显式转换到新目录，原目录只读）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py rescore --source-run eval-20261009 --run-id eval-20261009-r1 --reason "R1 返工：修复 F-01~F-08"
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py report --run-id eval-20261009-r1

# 5c) R2 返工重算（源目录 eval-20261009-r1 只读；R2 评分/报告写入新目录）
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py rescore --source-run eval-20261009-r1 --run-id eval-20261009-r2 --reason "R2 返工：F-R1-01~03 与 A-01~02"
.\.venv\Scripts\python.exe tests\run_quality_evaluation.py report --run-id eval-20261009-r2

# 6) 回归测试
.\.venv\Scripts\python.exe scripts\run_tests.py tests\test_quality_evaluation.py tests\test_evaluation_cases.py -q
.\.venv\Scripts\python.exe scripts\run_tests.py tests -q
```

真实采集需要 `DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY`（改写/生成/路由模型、向量与重排）。
缺凭据或外部服务不可用时保留已完成部分并在报告中标记未完成原因，不用桩结果填补真实成绩。
评测不新增外层自动重试；应用内部既有的重试行为（如重排服务的暂时性故障重试）原样保留。

## 4. 执行方式与数据隔离

- 端到端场景复用实际 LangGraph 编排（`app/graph.py` 的 prepare → retrieve → route → tools → generate），
  不另写评测专用问答链；两轮场景共享同一会话（thread_id = `eval-<run_id>-<scenario_id>`），
  不同场景会话相互隔离。
- 区分工具规划（`planned_tool_calls`）与实际执行（`tool_text`）；规划不作为调用成功证据，
  因此本阶段不输出「工具执行成功率」。
- route_only 场景只调用 `route()` 节点获取规划结果，不执行真实搜索，单独报告，不混入端到端成功率。
- 数据隔离（评测进程内完成，不增加应用配置项、不修改用户系统环境）：
  - 导入应用前把 `LAWRAG_DB_PATH` 指向 `logs/evaluation/<run_id>/lawrag.sqlite3`，
    并改绑 `app.db.DATABASE_PATH`、清空数据库/Checkpointer 连接缓存（顺序：graph → Checkpointer → metadata）。
  - Chroma 使用当前索引的隔离副本 `logs/evaluation/<run_id>/chroma_copy`：
    复制前后比对源目录快照（文件大小、mtime、SHA-256），检测并发入库；不一致则中止并标记
    「真实评测未执行」。绝不以只读方式打开真实 `chroma_db/`（E-059）。
  - 改绑 `app.ask.CHROMA_DIR`（模块级导入绑定）并清空 `load_index`/`get_bm25_index` 缓存。
  - 默认关闭 Langfuse 上报（进程内清空 `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`）。
  - 采集前后对真实 `chroma_db/`、`data/`、`kb/` 做只读快照并逐项比对，结果写入 `manifest.json`
    的 `real_dirs_unchanged`；结束后关闭本次 SQLite 连接，保留运行证据，不清理用户原有数据。

## 5. 运行产物（logs/evaluation/<run_id>/，不提交 Git）

| 文件 | 内容 |
| --- | --- |
| manifest.json | 代码 SHA、题集 hash、模型标识、检索策略、运行模式、隔离信息、真实目录快照、attempt 策略、时间 |
| results.jsonl | 每场景/每轮原始采集结果或错误：问题、改写问题（search_query）、检索条文及文本、回答、工具规划、工具输出（tool_text）、web_sources、错误信息；逐轮即时落盘 |
| reviews.json | 语义评分及评分依据（`report` 首次运行生成待填写模板，不覆盖已有记录） |
| report.md | 汇总、分组结果、失败案例、运行错误与未验证项 |

- 定向重跑会追加新 attempt 记录；汇总**始终使用每个 (scenario, turn) 的最新 attempt**
  并在报告中写明，不自动取最高分。
- 运行目录、数据库副本、密钥与原始模型输出均不提交 Git（`logs/` 在 .gitignore 中）。

## 6. 指标口径

所有指标同时报告：通过数/适用总数、失败数、运行错误数、待复核数、未采集数、不适用数。
运行错误与未采集不从分母消失、不冒充通过；语义项未完成复核时不得输出最终质量结论，
可报告「已复核部分成绩」但必须同时展示复核覆盖率。

### 6.1 检索 Hit@3（自动）

- 适用轮次：有 `expected_articles` 的问答图轮次；库外题、无检索目标题、route_only 不进分母。
- 命中：前三条（`citations` 前 3）中至少一条命中允许法条（标签格式「法律名 第X条」）。
- 逐题列出命中与缺失法条，避免「命中一条」掩盖多条文任务的证据缺口。

### 6.2 工具规划与参数（自动）

- 工具选择正确率：适用轮次中规划工具与预期一致的比例，**包括预期不调用的场景**
  （预期不调用时，出现任何 `planned_tool_calls` 即失败）。
- 参数正确率：需要调用计算器的适用轮次中，关键参数全部正确的比例。归一化与容差规则：
  - 数值：int/float，或去掉「元/年/月/个/逗号/空白」后可解析的字符串；绝对容差 0.01。
  - `kind`：去空白、全角＋转半角、大写后比对（N / N+1 / 2N）；计算器调用缺省 `kind` 视为默认值 N。
  - `query`（路由工具）：必须取自本轮问题（`{"match": "question"}`）。
  - 额外/多余参数一律视为编造参数，判错。
- 缺失调用、错误工具、缺失参数、编造参数均不计为正确。
- 两个联网路由场景（TOOL-WEB-01/02）单独报告；无实际执行证据，不输出「工具执行成功率」。

### 6.3 回答质量与引用支持（人工复核，评分表 + reviews.json，不新增 LLM 裁判）

逐轮复核项：要点覆盖（required_points 逐条）、实质性结论的引用支持（逐结论）、
禁止结论/编造检查（forbidden_claims 逐条）、拒答与边界处理（适用轮次）；
逐场景复核项：completion_criteria 判定。

- 引用支持率 = 已复核且有正确引用支持的实质性结论数 / 已复核的全部实质性结论数；
  需要依据但没有引用的结论不计入分子；无可评结论时为 N/A。
  「回答带有条号」和「接口返回 citations」都不能直接判为引用正确。
- 拒答与边界处理不做固定字符串匹配，检查：资料不足时是否明确说明不足；说明不足后是否仍
  给出无依据结论（边拒答边编造）；本可回答的库内题是否被错误拒答。
- 评分记录保留结果、理由、证据位置（results.jsonl 定位）及评分者身份。
  编码 agent 的语义评分只能标记为 `executor_preliminary`（执行者初评），不构成独立验收。

### 6.4 场景完成率（人工复核）

以场景冻结的 `completion_criteria` 判定；多轮场景按完整两轮结果判定。

## 7. 针对性测试覆盖（tests/test_quality_evaluation.py、tests/test_evaluation_cases.py）

不合法题集及重复 ID；预期字段不进入模型输入；工具名正确但关键参数错误；预期不调用却产生调用；
拒答语后仍含禁止结论的复核流转；citations 存在但不支持结论；错误与待复核结果的汇总；
两轮会话共享与跨场景隔离；保存结果后的纯离线汇总（断言不调用模型）；数据库与索引路径隔离
（含复制不一致即中止、真实目录前后快照）；最新 attempt 选择而非最佳成绩；语义判定只来自
复核记录（模板不做关键词判定伪装人工判断）。
R2 追加（均经真实 `report` 入口验证）：只改内容保留旧 hash 的参数化反例
（answer/citations/planned_tool_calls/tool_text/web_sources/search_query/status/error_type/error_message）、
缺 hash 须显式转换、R1 未改记录身份通过、冻结身份七类反例（改 expected_articles/annotation_status/
reviews.cases_sha256/manifest/conversion 各自的预期拦截或失效，失败路径核对证据文件 hash 未变、
既有报告未被覆盖，工作区题集改动不影响历史冻结运行）、待标注结论 supported=True/False/None 分流、
50=46+4 可核算关系（夹具数字）与三类未完成原因的报告文案。

## 8. 评测口径与产物（2026-10-09）

- F-01 会话隔离：每个场景每次 attempt 使用独立会话（`eval-<run_id>-<scenario>-a<attempt>-<nonce>`），
  同 attempt 内轮次共享会话；中断在首条结果落盘前也不会复用旧会话。
- F-01 attempt 组选择：按完整场景 attempt 组选取最新 attempt 再读取各轮；最新 attempt 不完整时
  如实显示缺失轮次（NOT_VERIFIED），不回退、不跨 attempt 拼接。
- F-02 评分绑定：reviews.json（v2）绑定冻结题集、场景、轮次、attempt 与结果内容 hash
  （`result_hash` 覆盖答案、引用、工具规划、tool_text、web_sources、search_query、状态与错误信息）；
  对照冻结定义校验评分项完整性（缺项/重复/替换/无依据 N/A 均无效）；绑定失配或评分无效 → 待答案复核；
  报告生成只读 reviews.json，不自动补 hash。
- F-03 冻结版本：collect 冻结题集快照（cases_frozen.json，按字节写入保证 hash 稳定）并逐 attempt 记录
  provenance（manifest 的 collection_passes 只追加不覆盖）；report 只读冻结快照，不读工作区题集；
  历史运行缺冻结快照时用 `rescore` 显式转换（核验来源 hash、记录转换过程与新旧评分 SHA），
  题集不一致时明确拒绝。
  - **原版重放**＝原题集 + 原评分规则解释旧成绩（原始运行目录保留原报告）；
  - **修正规则重算**＝原始采集结果 + 本次修正规则生成新报告（report_mode=rescore_with_repaired_rules，
    标明原运行身份、原始/新评分 SHA 与变更原因；不是原版成绩的无变化重放）。
- F-04 待标注复核：`needs_legal_review` 4 题（QA-LAB-03、QA-LAB-08、QA-CON-05、TOOL-CALC-04）的
  自动与语义评分全部计入“待标注复核”，不进可信 PASS/FAIL；明细表中的“试算”仅诊断用途。
- F-05 分母与状态：适用总数来自冻结定义（工具选择 46、参数 6、检索 37、graph 完成 38、route_only 2），
  状态互斥且满足 适用总数 = 通过 + 失败 + 运行错误 + 未采集 + 待标注复核 + 待答案复核；N/A 单列；
  已观察子集成绩单独命名并附覆盖率；graph 与 route_only 完成分开；任一必要轮次报错/缺失时场景不能 PASS。
- F-06 拒答两类指标：「应拒答/需说明不足场景的处理」与「可答题场景的错误拒答」分别统计，
  适用性由冻结 required_points 推导（无场景 ID 特判），判定来自人工复核记录，不用关键词替代语义判断。
- F-07 引用审核明细：分别报告已提取需证据结论总数、已审核数、支持数、不支持数、待审数
  （supported=None）与结论提取完整性未确认的回答数；已审核引用支持率与已提取结论审核覆盖率
  分开计算并与其他语义覆盖率分开展示；分母为零显示 N/A。
- F-08 完成判定：criteria_results 逐项对应冻结 completion_criteria；必要条件 FAIL/待审 → 整体不 PASS；
  完成 PASS 与轮次 FAIL 并存必须提供 reconciles_turn_failures 显式解释，否则评分无效；
  轮次复核无效时完成不得 PASS。

### 8.1 结果身份与评分有效性校验（2026-10-09）

- R2-01 结果身份三方校验：评分前按记录**当前内容**重算 `result_hash`，核验
  “保存 hash—实际内容 hash—复核绑定 hash”三方一致；内容失配的轮次与场景评分一律无效/待答案复核，
  不沿用旧 PASS，也不自动更新旧 hash 掩盖失配。缺 hash 的历史记录只能经 `rescore` 显式转换补算
  （来源记录于 `conversion.json` 的 `result_hash_missing_derived`）；已有的失配记录保留原 hash 并列入
  `conversion.identity_mismatches`，不重新包装为可信记录。
- R2-02 冻结身份核验：评分前比较 `cases_frozen.json` 当前字节 hash 与 `manifest`、`conversion` 中的
  冻结身份（字段含义见 2.4，原始文件 hash 与冻结 hash 不判等）；快照身份失配时**拒绝生成报告**
  （报错退出、不改写任何证据文件、不覆盖已有报告）；仅 `reviews.cases_sha256` 失配时，
  全部复核失效（待答案复核）并在报告中写明原因。
- R2-03 待标注引用分流：`needs_legal_review` 题的需证据结论全部计入“待标注复核”，
  执行者初评（supported=True/False/None）仅留诊断计数，不进可信已审/支持/不支持；
  覆盖率分母不排除待标注结论。可核算关系：
  `已提取 = 可信已审 + 可信标注下待答案复核 + 待标注复核`，`可信已审 = 支持 + 不支持`；
  三类未完成原因为：标注待审（结论数）、结论支持待审（结论数）、结论提取未确认（按回答数，另计）。
- A-01（TOOL-CALC-01 初评纠正）：第一条结论按回答末句原句重写为条件式结论，理由改为
  “答案原句 / 问题前提 / 法条依据”三方分列（R1 引用的“协商一致解除劳动合同的，用人单位应当……”
  句式在回答中不存在，已纠正）；按冻结条件复核后结论数量 2 条不变、评分结果不变，
  各条理由与证据位置逐句重写（见 `logs/evaluation/eval-20261009-r2/reviews.json` 的 `revision` 字段）。
- A-02（数字与身份口径纠正）：
  - 原始运行目录 `eval-20261009` 共 **15** 个普通文件（6 个顶层 + 9 个 `chroma_copy`），不是 16；
  - 原版 v1 报告 `coverage.items_total = items_reviewed = 143`，即 **143/143**，不是 143/138
    （可批评旧覆盖率混合了不同统计单位，但引用旧值必须准确）；
  - 语义评分仍为 `executor_preliminary`（执行者初评）；“待答案复核为 0”仅表示程序校验与初评
    覆盖完成，**不能**解释为独立法律复核完成。

## 9. 已知限制

- 语义质量结论依赖人工复核完成度；执行者初评不等于独立验收（REVIEW_PENDING）。
- route_only 只验证工具规划；联网检索与端到端联网任务不在本阶段。
- 缺参场景只验证「不编造、不无依据计算」，不验证结构化追问（尚未实现）。
- 规划层可见性止于应用输出的 `planned_tool_calls`（路由模型内部原始 tool_calls 被应用过滤，
  评测不穿透应用逻辑观察）。
- 本阶段不改 `app/` 业务代码、Prompt、路由规则、模型配置、检索参数与原有 55 题口径。
