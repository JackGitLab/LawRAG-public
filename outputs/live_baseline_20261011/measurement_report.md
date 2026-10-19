# LawRAG 请求性能与调用成本基线报告（R1 口径）

本报告由 `tests/run_performance_baseline.py report` 从运行目录已保存数据离线重建，不调用任何服务。
指标一律从原始 SSE 时间线重算；缺时间线的样本标 NOT_VERIFIED（不倒造事件时间线）。

## 1. 运行标识与版本绑定

- run_id：perf-20261011-live1
- 采集代码版本：git HEAD `fd75d225881956f876819df7f073b0f00b546963`（采集时工作区无未提交改动）
- 采集文件哈希：app/ask.py = `4a75381b56ed4d143bb71deb04cc117d16bb4010320d0b82dc1139673b2d1cd9`
- 采集文件哈希：app/config.py = `30eca98a5bec8a5c039ea0add634fa2851c95a650d0a8bb0ab9daf045e2f0658`
- 采集文件哈希：app/graph.py = `88befffa619e3e60d874386479114f6c26fb4af4e20660c92e66cdb2ee025bcc`
- 采集文件哈希：app/main.py = `295d50891e21ef35e1ed340a97e26e81e538a2315c6038cfccf9458858de2e6b`
- 采集文件哈希：app/observability.py = `b40572145158e3b00b180d662fd22e36011ccc86497de3f8e29fea8ea2ec67c2`
- 采集文件哈希：outputs\live_baseline_20261011\pricing.json = `2ee595448bb57cbc0ea8dc05fb6c1def35f9b75cc104deb03533ae6fffe34212`
- 采集文件哈希：outputs\live_baseline_20261011\run_capture.py = `5cb7cd387cc87c624475de725221d42878614582f6a08df2faa22e37fa5c8d2d`
- 采集文件哈希：tests/performance_cases.json = `4e8f5e1cab7380483014a9150e8df99b1498516ee12d10cd05d0b78e53bfaf02`
- 采集文件哈希：tests/run_performance_baseline.py = `2e1d146b32116e55d496fe0c8d94810809ca904c863dc8c54a0580ad422aaa13`
- 报告生成代码版本：tests/run_performance_baseline.py = `2e1d146b32116e55d496fe0c8d94810809ca904c863dc8c54a0580ad422aaa13`
- 说明：采集代码版本与报告生成代码版本独立记录；本报告不含重建时刻（保证同输入重建字节一致）。
- 场景文件哈希：`e9516e04baef085824f5593f081a379b7dde9c8330a5b7f7516ac6034e610b9e`
- 场景冻结日期：2026-10-10
- 采集时间：2026-10-11T03:17:45.791+00:00 ~ 2026-10-11T03:21:26.430+00:00
- 模型：chat=deepseek-chat，router=deepseek-flash，embedding=BAAI/bge-m3，rerank=BAAI/bge-reranker-v2-m3
- 检索策略：hybrid_rerank
- Langfuse：disabled（测量期间统一关闭外部上报）
- 服务启动耗时（另列，不属请求指标）：5.619 秒

## 2. 测量方法与指标定义

- 首段可见回答耗时：客户端开始发送请求 → 第一个非空回答文本事件（token）。`session` 事件不是回答首字。
- 请求完成耗时：客户端开始发送请求 → 唯一 `done`。
- 失败耗时：请求开始 → 错误事件或连接异常。
- 成功条件：无连接异常、无 error 事件、恰好 1 个 done、有非空回答且首段在 done 之前、done 后无非空文本；“done 之后”按原始事件接收顺序判断，同刻事件按保存顺序定先后，时间戳只用于计时。
- **成功指标（first_text_ms / done_ms）只属于满足成功条件的请求**；失败请求即使已收到 token 或 done，其时间只保留为原始观察值（observed_*），不进入成功统计。HTTP 200 不等于任务成功。
- 场景耗时：一次完整场景执行（多轮共一个任务）各轮请求耗时之和；只有计划规定的每一轮恰好出现一次、全部必需轮次成功且路径符合预期才计为完整成功任务（轮次身份与唯一性），部分成功轮次、重复轮次、缺轮或越界轮次均不相加为完整场景耗时。
- 计划任务/计划请求来自 manifest 冻结计划（场景、warmup_passes、formal_reps、必需轮次）；整体表、逐轮表、场景表与费用说明使用同一份计划口径，未执行任务保留展示，不自动补跑。实际记录按槽位身份 (scenario_id, phase, rep, turn_index) 与冻结计划匹配：恰好一条才算计划内成功，无记录为未执行，同槽位多条为重复异常；计划外/重复/越界记录单列保留身份，不填补缺失槽位，不进计划内成功耗时与完整成功任务分母（其可核实费用仍计费用分子）。
- 服务端节点耗时：prepare/retrieve/route/tools/clarify/generate 各节点实测耗时（服务端时钟）。
- 工具执行耗时：每次实际工具调用开始→结束（含失败调用）；mcp_init 为联网工具初始化。
- 确定性追问（clarify）整段输出的首段耗时是确定性输出耗时，不是模型首 token 时间；逐样本标注 first_text_kind。
- SSE 文本事件数 ≠ 模型 token 数；token 用量只采用供应商返回的 usage，按字段判已知/未知。
- 服务端分段与客户端时间分别统计，两者时钟不相减。node 层与 llm/tool/retrieval 调用层为包含关系，不跨层相加。
- 统计口径：客户端成功耗时表只含成功请求；节点/调用/瓶颈表只用**正式样本**（含失败请求的服务端事实，标注于表头）；预热单列；调用与费用另给“含预热”的整次采集总账。

## 3. 客户端指标：逐样本（成功指标与原始观察分列）

| 请求 | 阶段 | 轮 | HTTP | 状态 | 首段(ms) | 完成(ms) | 失败(ms) | 原始观察首段/完成(ms) | done数 | 文本事件数 | 原始时间线 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| PERF-LIB-01-warmup1-t1 | warmup | 1 | 200 | ok | 6511.926 | 6789.931 | 缺失 | 6511.926/6789.931 | 1 | 79 | 可重算 |
| PERF-MULTI-01-warmup1-t1 | warmup | 1 | 200 | ok | 4256.146 | 4728.038 | 缺失 | 4256.146/4728.038 | 1 | 92 | 可重算 |
| PERF-MULTI-01-warmup1-t2 | warmup | 2 | 200 | ok | 7297.731 | 7663.3 | 缺失 | 7297.731/7663.3 | 1 | 71 | 可重算 |
| PERF-CALC-01-warmup1-t1 | warmup | 1 | 200 | ok | 5378.667 | 5378.912 | 缺失 | 5378.667/5378.912 | 1 | 1 | 可重算 |
| PERF-CALC-01-warmup1-t2 | warmup | 2 | 200 | ok | 21003.62 | 21387.748 | 缺失 | 21003.62/21387.748 | 1 | 74 | 可重算 |
| PERF-WEB-01-warmup1-t1 | warmup | 1 | 200 | ok | 9999.592 | 10100.518 | 缺失 | 9999.592/10100.518 | 1 | 5 | 可重算 |
| PERF-LIB-01-formal1-t1 | formal | 1 | 200 | ok | 5029.12 | 5296.606 | 缺失 | 5029.12/5296.606 | 1 | 79 | 可重算 |
| PERF-MULTI-01-formal1-t1 | formal | 1 | 200 | ok | 4172.907 | 4482.216 | 缺失 | 4172.907/4482.216 | 1 | 98 | 可重算 |
| PERF-MULTI-01-formal1-t2 | formal | 2 | 200 | ok | 6531.472 | 6857.759 | 缺失 | 6531.472/6857.759 | 1 | 58 | 可重算 |
| PERF-CALC-01-formal1-t1 | formal | 1 | 200 | ok | 5278.631 | 5278.9 | 缺失 | 5278.631/5278.9 | 1 | 1 | 可重算 |
| PERF-CALC-01-formal1-t2 | formal | 2 | 200 | ok | 33664.459 | 34231.237 | 缺失 | 33664.459/34231.237 | 1 | 78 | 可重算 |
| PERF-WEB-01-formal1-t1 | formal | 1 | 200 | ok | 9084.365 | 9156.639 | 缺失 | 9084.365/9156.639 | 1 | 5 | 可重算 |
| PERF-LIB-01-formal2-t1 | formal | 1 | 200 | ok | 5316.83 | 5761.485 | 缺失 | 5316.83/5761.485 | 1 | 107 | 可重算 |
| PERF-MULTI-01-formal2-t1 | formal | 1 | 200 | ok | 3893.59 | 4250.556 | 缺失 | 3893.59/4250.556 | 1 | 98 | 可重算 |
| PERF-MULTI-01-formal2-t2 | formal | 2 | 200 | ok | 7195.718 | 7446.723 | 缺失 | 7195.718/7446.723 | 1 | 57 | 可重算 |
| PERF-CALC-01-formal2-t1 | formal | 1 | 200 | ok | 5105.659 | 5105.988 | 缺失 | 5105.659/5105.988 | 1 | 1 | 可重算 |
| PERF-CALC-01-formal2-t2 | formal | 2 | 200 | ok | 10762.265 | 11132.797 | 缺失 | 10762.265/11132.797 | 1 | 74 | 可重算 |
| PERF-WEB-01-formal2-t1 | formal | 1 | 200 | ok | 8887.015 | 8990.922 | 缺失 | 8887.015/8990.922 | 1 | 5 | 可重算 |
| PERF-LIB-01-formal3-t1 | formal | 1 | 200 | ok | 5830.811 | 6202.337 | 缺失 | 5830.811/6202.337 | 1 | 79 | 可重算 |
| PERF-MULTI-01-formal3-t1 | formal | 1 | 200 | ok | 4302.983 | 4691.313 | 缺失 | 4302.983/4691.313 | 1 | 98 | 可重算 |
| PERF-MULTI-01-formal3-t2 | formal | 2 | 200 | ok | 8541.505 | 9015.134 | 缺失 | 8541.505/9015.134 | 1 | 98 | 可重算 |
| PERF-CALC-01-formal3-t1 | formal | 1 | 200 | ok | 10289.007 | 10289.37 | 缺失 | 10289.007/10289.37 | 1 | 1 | 可重算 |
| PERF-CALC-01-formal3-t2 | formal | 2 | 200 | ok | 9957.192 | 10273.487 | 缺失 | 9957.192/10273.487 | 1 | 74 | 可重算 |
| PERF-WEB-01-formal3-t1 | formal | 1 | 200 | ok | 10298.625 | 10380.092 | 缺失 | 10298.625/10380.092 | 1 | 5 | 可重算 |

## 4. 样本与任务口径

- 计划样本：预热 6 + 正式 18 = 24（冻结计划槽位，**计划分母不因失败或计划外记录改变**）；计划槽位结果：成功 24，失败 0，重复异常 0，未执行 0，成功率 24/24。
- 实际请求 24，成功 24，失败 0（物理计数）；计划内唯一匹配 24 条，计划外/重复异常记录 0 条（计划外 0、重复 0），单列如下，不填补缺失槽位。
- 计划外/重复异常记录：无。

- 任务口径（一次完整场景执行 = 一个任务，多轮共一个任务；**计划任务来自 manifest 冻结计划**，不随实际记录增减）：预热任务：计划 4，实际执行 4，完整成功 4，未执行 0，已执行未完成或失败 0；正式任务：计划 12，实际执行 12，完整成功 12，未执行 0，已执行未完成或失败 0；计划外任务 0 个（单列，不进计划口径）。

| 任务 | 阶段 | 计划轮次 | 实际轮次 | 状态 | 请求耗时合计(ms，仅事实) | 未完成原因与实际记录 |
| --- | --- | ---: | ---: | --- | ---: | --- |
| PERF-CALC-01-formal1 | formal | 2 | 2 | 完整成功 | 39510.841 | -；实际记录：['PERF-CALC-01-formal1-t1(T1,ok)', 'PERF-CALC-01-formal1-t2(T2,ok)'] |
| PERF-CALC-01-formal2 | formal | 2 | 2 | 完整成功 | 16239.369 | -；实际记录：['PERF-CALC-01-formal2-t1(T1,ok)', 'PERF-CALC-01-formal2-t2(T2,ok)'] |
| PERF-CALC-01-formal3 | formal | 2 | 2 | 完整成功 | 20563.51 | -；实际记录：['PERF-CALC-01-formal3-t1(T1,ok)', 'PERF-CALC-01-formal3-t2(T2,ok)'] |
| PERF-CALC-01-warmup1 | warmup | 2 | 2 | 完整成功 | 26767.127 | -；实际记录：['PERF-CALC-01-warmup1-t1(T1,ok)', 'PERF-CALC-01-warmup1-t2(T2,ok)'] |
| PERF-LIB-01-formal1 | formal | 1 | 1 | 完整成功 | 5296.932 | -；实际记录：['PERF-LIB-01-formal1-t1(T1,ok)'] |
| PERF-LIB-01-formal2 | formal | 1 | 1 | 完整成功 | 5761.865 | -；实际记录：['PERF-LIB-01-formal2-t1(T1,ok)'] |
| PERF-LIB-01-formal3 | formal | 1 | 1 | 完整成功 | 6202.563 | -；实际记录：['PERF-LIB-01-formal3-t1(T1,ok)'] |
| PERF-LIB-01-warmup1 | warmup | 1 | 1 | 完整成功 | 6790.171 | -；实际记录：['PERF-LIB-01-warmup1-t1(T1,ok)'] |
| PERF-MULTI-01-formal1 | formal | 2 | 2 | 完整成功 | 11340.422 | -；实际记录：['PERF-MULTI-01-formal1-t1(T1,ok)', 'PERF-MULTI-01-formal1-t2(T2,ok)'] |
| PERF-MULTI-01-formal2 | formal | 2 | 2 | 完整成功 | 11697.721 | -；实际记录：['PERF-MULTI-01-formal2-t1(T1,ok)', 'PERF-MULTI-01-formal2-t2(T2,ok)'] |
| PERF-MULTI-01-formal3 | formal | 2 | 2 | 完整成功 | 13706.976 | -；实际记录：['PERF-MULTI-01-formal3-t1(T1,ok)', 'PERF-MULTI-01-formal3-t2(T2,ok)'] |
| PERF-MULTI-01-warmup1 | warmup | 2 | 2 | 完整成功 | 12391.84 | -；实际记录：['PERF-MULTI-01-warmup1-t1(T1,ok)', 'PERF-MULTI-01-warmup1-t2(T2,ok)'] |
| PERF-WEB-01-formal1 | formal | 1 | 1 | 完整成功 | 9156.96 | -；实际记录：['PERF-WEB-01-formal1-t1(T1,ok)'] |
| PERF-WEB-01-formal2 | formal | 1 | 1 | 完整成功 | 8991.222 | -；实际记录：['PERF-WEB-01-formal2-t1(T1,ok)'] |
| PERF-WEB-01-formal3 | formal | 1 | 1 | 完整成功 | 10380.401 | -；实际记录：['PERF-WEB-01-formal3-t1(T1,ok)'] |
| PERF-WEB-01-warmup1 | warmup | 1 | 1 | 完整成功 | 10100.818 | -；实际记录：['PERF-WEB-01-warmup1-t1(T1,ok)'] |

### 4.1 正式场景耗时与首段/完成（仅计划内完整成功任务与计划槽位成功请求；计划列来自冻结计划）

| 场景×轮次 | 指标 | 成功n | 计划 | 失败/异常 | 未执行 | min | median | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PERF-LIB-01 T1 | 首段可见回答耗时(ms) | 3 | 3 | 0 | 0 | 5029.12 | 5316.83 | 5830.811 |
| PERF-LIB-01 T1 | 请求完成耗时(ms) | 3 | 3 | 0 | 0 | 5296.606 | 5761.485 | 6202.337 |
| PERF-LIB-01 | 场景耗时(ms) | 3 | 3 | 0 | 0 | 5296.932 | 5761.865 | 6202.563 |
| PERF-MULTI-01 T1 | 首段可见回答耗时(ms) | 3 | 3 | 0 | 0 | 3893.59 | 4172.907 | 4302.983 |
| PERF-MULTI-01 T1 | 请求完成耗时(ms) | 3 | 3 | 0 | 0 | 4250.556 | 4482.216 | 4691.313 |
| PERF-MULTI-01 T2 | 首段可见回答耗时(ms) | 3 | 3 | 0 | 0 | 6531.472 | 7195.718 | 8541.505 |
| PERF-MULTI-01 T2 | 请求完成耗时(ms) | 3 | 3 | 0 | 0 | 6857.759 | 7446.723 | 9015.134 |
| PERF-MULTI-01 | 场景耗时(ms) | 3 | 3 | 0 | 0 | 11340.422 | 11697.721 | 13706.976 |
| PERF-CALC-01 T1 | 首段可见回答耗时(ms) | 3 | 3 | 0 | 0 | 5105.659 | 5278.631 | 10289.007 |
| PERF-CALC-01 T1 | 请求完成耗时(ms) | 3 | 3 | 0 | 0 | 5105.988 | 5278.9 | 10289.37 |
| PERF-CALC-01 T2 | 首段可见回答耗时(ms) | 3 | 3 | 0 | 0 | 9957.192 | 10762.265 | 33664.459 |
| PERF-CALC-01 T2 | 请求完成耗时(ms) | 3 | 3 | 0 | 0 | 10273.487 | 11132.797 | 34231.237 |
| PERF-CALC-01 | 场景耗时(ms) | 3 | 3 | 0 | 0 | 16239.369 | 20563.51 | 39510.841 |
| PERF-WEB-01 T1 | 首段可见回答耗时(ms) | 3 | 3 | 0 | 0 | 8887.015 | 9084.365 | 10298.625 |
| PERF-WEB-01 T1 | 请求完成耗时(ms) | 3 | 3 | 0 | 0 | 8990.922 | 9156.639 | 10380.092 |
| PERF-WEB-01 | 场景耗时(ms) | 3 | 3 | 0 | 0 | 8991.222 | 9156.96 | 10380.401 |

样本量说明：每场景正式重复 3 次，仅用于初步基线；**不发布 P95（样本不足）**，尾延迟未覆盖。
预热样本单列（下表），不进入上表任何统计，也不称为严格冷启动测试：

| 请求 | 首段(ms) | 完成(ms) | 状态 |
| --- | ---: | ---: | --- |
| PERF-LIB-01-warmup1-t1 | 6511.926 | 6789.931 | ok |
| PERF-MULTI-01-warmup1-t1 | 4256.146 | 4728.038 | ok |
| PERF-MULTI-01-warmup1-t2 | 7297.731 | 7663.3 | ok |
| PERF-CALC-01-warmup1-t1 | 5378.667 | 5378.912 | ok |
| PERF-CALC-01-warmup1-t2 | 21003.62 | 21387.748 | ok |
| PERF-WEB-01-warmup1-t1 | 9999.592 | 10100.518 | ok |

## 5. 服务端分段与调用量

### 5.1 正式样本（只用正式阶段观测；含失败请求的服务端事实，与客户端成功表口径不同，不混用）

#### 节点层（node，含其内部调用，不与调用层相加）

| 节点 | 次数 | 失败 | 总耗时(ms) |
| --- | ---: | ---: | ---: |
| clarify | 3 | 0 | 0.089 |
| generate | 15 | 0 | 29381.796 |
| prepare | 18 | 0 | 10549.614 |
| retrieve | 18 | 0 | 67613.749 |
| route | 18 | 0 | 40171.99 |
| tools | 6 | 0 | 10811.061 |

#### 调用层（LLM 逻辑调用，单点计量不重复计费）

| 用途 | 次数 | 失败 | 总耗时(ms) | usage已知 | usage缺失 |
| --- | ---: | ---: | ---: | ---: | ---: |
| generate | 15 | 0 | 29369.1 | 15 | 0 |
| rewrite | 6 | 0 | 10545.741 | 6 | 0 |
| router | 6 | 0 | 40167.554 | 6 | 0 |

#### 检索调用层（向量/BM25/重排；BM25 为进程内计算，无外部服务调用）

| 步骤 | 次数 | 失败 | 总耗时(ms) | 尝试次数 |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 18 | 0 | 14.144 | - |
| rerank | 18 | 0 | 38892.291 | 18 |
| vector | 18 | 0 | 28680.284 | - |

#### 工具执行层（含失败调用与联网初始化）

| 工具 | 次数 | 失败 | 总耗时(ms) |
| --- | ---: | ---: | ---: |
| calculate_labor_compensation | 3 | 0 | 6.799 |
| mcp_init | 3 | 0 | 0.005 |
| verify_statute_status | 3 | 0 | 10799.778 |

### 5.2 预热样本（仅预热，不进入任何正式统计）

#### 节点层（node，含其内部调用，不与调用层相加）

| 节点 | 次数 | 失败 | 总耗时(ms) |
| --- | ---: | ---: | ---: |
| clarify | 1 | 0 | 0.026 |
| generate | 5 | 0 | 10957.291 |
| prepare | 6 | 0 | 3461.125 |
| retrieve | 6 | 0 | 20165.533 |
| route | 6 | 0 | 16734.836 |
| tools | 2 | 0 | 4551.464 |

#### 调用层（LLM 逻辑调用，单点计量不重复计费）

| 用途 | 次数 | 失败 | 总耗时(ms) | usage已知 | usage缺失 |
| --- | ---: | ---: | ---: | ---: | ---: |
| generate | 5 | 0 | 10953.116 | 5 | 0 |
| rewrite | 2 | 0 | 3460.02 | 2 | 0 |
| router | 2 | 0 | 16731.796 | 2 | 0 |

#### 检索调用层（向量/BM25/重排；BM25 为进程内计算，无外部服务调用）

| 步骤 | 次数 | 失败 | 总耗时(ms) | 尝试次数 |
| --- | ---: | ---: | ---: | ---: |
| bm25 | 6 | 0 | 4.399 | - |
| rerank | 6 | 0 | 11419.307 | 6 |
| vector | 6 | 0 | 8732.712 | - |

#### 工具执行层（含失败调用与联网初始化）

| 工具 | 次数 | 失败 | 总耗时(ms) |
| --- | ---: | ---: | ---: |
| calculate_labor_compensation | 1 | 0 | 2.591 |
| mcp_init | 1 | 0 | 723.549 |
| verify_statute_status | 1 | 0 | 3823.45 |

### 5.3 整次采集审计总账（含预热，仅供对账）

- 调用总数（含预热）：LLM 36 次（正式 27 + 预热 9 + 未归属 0 条观测）；检索 72 次；工具 12 次。
- 预热＋正式＝整次采集总账可核对；上表正式/预热口径之和与本总账一致。
- 统计口径：改写（rewrite）、路由（router）、生成（generate）分别统计；向量化、重排、联网搜索均在调用清单中；库内部不可观测的网络尝试不猜测；重排尝试次数为实际可观察值，历史记录缺 attempts 记 UNKNOWN，不按 0 处理。

## 6. token 用量与费用覆盖

口径：费用只使用本次运行 manifest 冻结的单价与模型映射（不使用当前代码默认价格）；**计费逐调用判断**（该次调用所需 usage 与费率完整才计费），再按模型汇总，同模型的未知调用不抹掉其他调用的已知费用；usage 按字段判已知/未知，缺失不填零；费用合计只是**已知部分费用**，存在未知调用时不等于完整费用。

### 6.1 正式样本费用

| 模型 | 调用 | 计费调用 | 输入tokens | 输出tokens | 缓存命中tokens | 估算费用 | 价格依据 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek-chat | 21 | 0 | 14485 | 1194 | 10997 | 未知 | unknown |
| deepseek-flash | 6 | 6 | 3996 | 6657 | 2688 | CNY 0.02799 | frozen_verified |
| BAAI/bge-m3（vector） | 18 | 0 | None | None | None | 未知 | unknown |
| BAAI/bge-reranker-v2-m3（rerank） | 18 | 0 | None | None | None | 未知 | unknown |
| calculate_labor_compensation（按次） | 3 | 0 | None | None | None | 未知 | missing |
| verify_statute_status（按次） | 3 | 0 | None | None | None | 未知 | unknown |

- 已知部分费用合计：**CNY 0.02799**（按冻结公开单价估算（非实际扣费）；仅为可核实的已知部分，不等于完整费用）
- 未知项（不填零）：
  - 模型 deepseek-chat：21 次调用费用未知（单次调用缺多个字段只计一次）——模型 deepseek-chat 缺少可核验的单价（价格状态 unknown）（21 次）
  - BAAI/bge-m3（vector，18 次）：供应商未返回 token 用量，按量费用未知（公开免费档仅是估算假设，不证明账户费用为零）
  - BAAI/bge-reranker-v2-m3（rerank，18 次）：供应商未返回 token 用量，按量费用未知（公开免费档仅是估算假设，不证明账户费用为零）
  - calculate_labor_compensation（3 次）：单次价格不可核验（price_status=missing），费用未知，不以 0 表示
  - verify_statute_status（3 次）：单次价格不可核验（price_status=unknown），费用未知，不以 0 表示
  - 所有已计费用均为按冻结公开单价估算，未与实际账单核对；折扣、套餐抵扣不计

### 6.2 预热样本费用

| 模型 | 调用 | 计费调用 | 输入tokens | 输出tokens | 缓存命中tokens | 估算费用 | 价格依据 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek-chat | 7 | 0 | 4823 | 391 | 3707 | 未知 | unknown |
| deepseek-flash | 2 | 2 | 1332 | 3081 | 896 | CNY 0.012778 | frozen_verified |
| BAAI/bge-m3（vector） | 6 | 0 | None | None | None | 未知 | unknown |
| BAAI/bge-reranker-v2-m3（rerank） | 6 | 0 | None | None | None | 未知 | unknown |
| calculate_labor_compensation（按次） | 1 | 0 | None | None | None | 未知 | missing |
| verify_statute_status（按次） | 1 | 0 | None | None | None | 未知 | unknown |

- 已知部分费用合计：**CNY 0.012778**（按冻结公开单价估算（非实际扣费）；仅为可核实的已知部分，不等于完整费用）
- 未知项（不填零）：
  - 模型 deepseek-chat：7 次调用费用未知（单次调用缺多个字段只计一次）——模型 deepseek-chat 缺少可核验的单价（价格状态 unknown）（7 次）
  - BAAI/bge-m3（vector，6 次）：供应商未返回 token 用量，按量费用未知（公开免费档仅是估算假设，不证明账户费用为零）
  - BAAI/bge-reranker-v2-m3（rerank，6 次）：供应商未返回 token 用量，按量费用未知（公开免费档仅是估算假设，不证明账户费用为零）
  - calculate_labor_compensation（1 次）：单次价格不可核验（price_status=missing），费用未知，不以 0 表示
  - verify_statute_status（1 次）：单次价格不可核验（price_status=unknown），费用未知，不以 0 表示
  - 所有已计费用均为按冻结公开单价估算，未与实际账单核对；折扣、套餐抵扣不计

### 6.3 整次采集费用（含预热）

| 模型 | 调用 | 计费调用 | 输入tokens | 输出tokens | 缓存命中tokens | 估算费用 | 价格依据 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| deepseek-chat | 28 | 0 | 19308 | 1585 | 14704 | 未知 | unknown |
| deepseek-flash | 8 | 8 | 5328 | 9738 | 3584 | CNY 0.040768 | frozen_verified |
| BAAI/bge-m3（vector） | 24 | 0 | None | None | None | 未知 | unknown |
| BAAI/bge-reranker-v2-m3（rerank） | 24 | 0 | None | None | None | 未知 | unknown |
| calculate_labor_compensation（按次） | 4 | 0 | None | None | None | 未知 | missing |
| verify_statute_status（按次） | 4 | 0 | None | None | None | 未知 | unknown |

- 已知部分费用合计：**CNY 0.040768**（按冻结公开单价估算（非实际扣费）；仅为可核实的已知部分，不等于完整费用）
- 未知项（不填零）：
  - 模型 deepseek-chat：28 次调用费用未知（单次调用缺多个字段只计一次）——模型 deepseek-chat 缺少可核验的单价（价格状态 unknown）（28 次）
  - BAAI/bge-m3（vector，24 次）：供应商未返回 token 用量，按量费用未知（公开免费档仅是估算假设，不证明账户费用为零）
  - BAAI/bge-reranker-v2-m3（rerank，24 次）：供应商未返回 token 用量，按量费用未知（公开免费档仅是估算假设，不证明账户费用为零）
  - calculate_labor_compensation（4 次）：单次价格不可核验（price_status=missing），费用未知，不以 0 表示
  - verify_statute_status（4 次）：单次价格不可核验（price_status=unknown），费用未知，不以 0 表示
  - 所有已计费用均为按冻结公开单价估算，未与实际账单核对；折扣、套餐抵扣不计

### 6.4 每成功任务费用（正式）

- 正式完整成功任务数：12（分母；计划正式任务 12）；预热任务 4 另计；计划外正式任务 0 个单列（不进计划口径，其已发生、可核实的费用仍计入分子）。
- 正式每成功任务已知费用：**0.0023325**。正式每成功任务已知费用 = 正式范围全部已知费用（含失败及计划外请求已发生的可核实费用）÷ 正式计划内完整成功任务数；仅为已知费用部分，不等同实际扣费；成功任务数为 0 时为 N/A。

### 6.5 单价依据（冻结于 manifest，含来源摘录与适用条件）

- DeepSeek / deepseek-flash：元/百万tokens，币种 CNY，状态 frozen_verified，来源 https://api-docs.deepseek.com/zh-cn/quick_start/pricing/（2026-10-11 抓取）。
  - 来源摘录：2026-10-11 浏览工具读取官方页面：Flash 空闲费率为缓存命中输入 0.02、未命中输入 1、输出 4 元/百万 tokens；高峰对应 0.04、2、8。
  - 适用条件：北京时间周末全日空闲；本次授权采集日期 2026-10-11 为周日。公开目录价估算，非账户实际扣费。
  - 备注：来源为本次浏览工具读取内容的数值摘录；未获取账户账单。
- DeepSeek / deepseek-chat：元/百万tokens，币种 CNY，状态 unknown，来源 https://api-docs.deepseek.com/zh-cn/quick_start/pricing/（2026-10-11 抓取）。
  - 来源摘录：当前读取页面列出 deepseek-flash 与 deepseek-v4-pro，没有确认 deepseek-chat 的现行计费映射。
  - 适用条件：保留应用实际模型配置；不把历史别名价格自动套到本次调用。
- SiliconFlow / BAAI/bge-m3：元/百万tokens，币种 CNY，状态 unknown，来源 https://siliconflow.cn/pricing（2026-10-11 抓取）。
  - 来源摘录：本次读取公开价格页，未取得足以核验此模型账户价格与 usage 的证据。
  - 适用条件：历史公开免费档不代表当前账户已核实零费用。
- SiliconFlow / BAAI/bge-reranker-v2-m3：元/百万tokens，币种 CNY，状态 unknown，来源 https://siliconflow.cn/pricing（2026-10-11 抓取）。
  - 来源摘录：本次读取公开价格页，未取得足以核验此模型账户价格与 usage 的证据。
  - 适用条件：历史公开免费档不代表当前账户已核实零费用。
- Bocha / web-search：元/次，币种 CNY，状态 unknown，来源 https://open.bochaai.com/（2026-10-11 抓取）。
  - 来源摘录：公开入口可读；本账户的单次有效费率未取得。
  - 适用条件：费用未知，不能记为零。
- Langfuse 可见的 LLM token 不能代表全部服务成本（不含向量、重排、联网搜索与本地工具）。

## 7. 预期路径与实际路径核对

路径符合预期只表示完成了指定技术流程，不代表独立法律质量审核通过。

- PERF-LIB-01-warmup1-t1（warmup）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=[]，实际 tools=[]
- PERF-LIB-01-formal1-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=[]，实际 tools=[]
- PERF-LIB-01-formal2-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=[]，实际 tools=[]
- PERF-LIB-01-formal3-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=[]，实际 tools=[]
- PERF-MULTI-01-warmup1-t1（warmup）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=[]，实际 tools=[]
- PERF-MULTI-01-warmup1-t2（warmup）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['rewrite', 'generate']，实际 llm=['rewrite', 'generate']；预期 tools=[]，实际 tools=[]
- PERF-MULTI-01-formal1-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=[]，实际 tools=[]
- PERF-MULTI-01-formal1-t2（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['rewrite', 'generate']，实际 llm=['rewrite', 'generate']；预期 tools=[]，实际 tools=[]
- PERF-MULTI-01-formal2-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=[]，实际 tools=[]
- PERF-MULTI-01-formal2-t2（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['rewrite', 'generate']，实际 llm=['rewrite', 'generate']；预期 tools=[]，实际 tools=[]
- PERF-MULTI-01-formal3-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=[]，实际 tools=[]
- PERF-MULTI-01-formal3-t2（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'generate']；预期 llm=['rewrite', 'generate']，实际 llm=['rewrite', 'generate']；预期 tools=[]，实际 tools=[]
- PERF-CALC-01-warmup1-t1（warmup）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'clarify']，实际 nodes=['prepare', 'retrieve', 'route', 'clarify']；预期 llm=['router']，实际 llm=['router']；预期 tools=[]，实际 tools=[]
- PERF-CALC-01-warmup1-t2（warmup）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']；预期 llm=['rewrite', 'router', 'generate']，实际 llm=['rewrite', 'router', 'generate']；预期 tools=['calculate_labor_compensation']，实际 tools=['calculate_labor_compensation']
- PERF-CALC-01-formal1-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'clarify']，实际 nodes=['prepare', 'retrieve', 'route', 'clarify']；预期 llm=['router']，实际 llm=['router']；预期 tools=[]，实际 tools=[]
- PERF-CALC-01-formal1-t2（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']；预期 llm=['rewrite', 'router', 'generate']，实际 llm=['rewrite', 'router', 'generate']；预期 tools=['calculate_labor_compensation']，实际 tools=['calculate_labor_compensation']
- PERF-CALC-01-formal2-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'clarify']，实际 nodes=['prepare', 'retrieve', 'route', 'clarify']；预期 llm=['router']，实际 llm=['router']；预期 tools=[]，实际 tools=[]
- PERF-CALC-01-formal2-t2（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']；预期 llm=['rewrite', 'router', 'generate']，实际 llm=['rewrite', 'router', 'generate']；预期 tools=['calculate_labor_compensation']，实际 tools=['calculate_labor_compensation']
- PERF-CALC-01-formal3-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'clarify']，实际 nodes=['prepare', 'retrieve', 'route', 'clarify']；预期 llm=['router']，实际 llm=['router']；预期 tools=[]，实际 tools=[]
- PERF-CALC-01-formal3-t2（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']；预期 llm=['rewrite', 'router', 'generate']，实际 llm=['rewrite', 'router', 'generate']；预期 tools=['calculate_labor_compensation']，实际 tools=['calculate_labor_compensation']
- PERF-WEB-01-warmup1-t1（warmup）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=['verify_statute_status']，实际 tools=['verify_statute_status']
- PERF-WEB-01-formal1-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=['verify_statute_status']，实际 tools=['verify_statute_status']
- PERF-WEB-01-formal2-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=['verify_statute_status']，实际 tools=['verify_statute_status']
- PERF-WEB-01-formal3-t1（formal）：路径一致；预期 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']，实际 nodes=['prepare', 'retrieve', 'route', 'tools', 'generate']；预期 llm=['generate']，实际 llm=['generate']；预期 tools=['verify_statute_status']，实际 tools=['verify_statute_status']

## 8. 失败、样本量与限制

- 共 24 个 HTTP 请求：预热 6 个，正式 18 个；失败/异常请求 0 个（全部保留，未补跑、未剔除；计划分母 24 不因失败减少）。
- 观测记录共 224 条（按 seq 去重后），未归属到任何请求窗口的 0 条（服务启动/索引加载等，不进入正式/预热统计）。
- 服务端分段计时来自应用内观测（env 门控、fail-open）；异常路径同样记录，失败调用保留。
- 客户端与服务端时钟分别统计，未做跨机器时钟相减；时间归属按墙钟窗口，串行执行下无重叠。
- 检索内部（向量召回内部步骤、BM25 分词）未进一步拆分。
- 3 次重复不足以稳定描述尾延迟，不发布 P95；不开展并发压测，不推断吞吐量或生产容量。
- 费用为按冻结公开单价估算，未核实账单、折扣与套餐；公开免费档不证明账户费用为零。
- 本报告只陈述测量事实与统计；推断与优化候选见 docs/PERFORMANCE.md，未验证项均已标注。

## 9. 瓶颈分析（正式样本，事实排序，不含优化承诺）

正式样本节点层耗时占比（与 5.1 同一集合；预热不参与）：

| 节点 | 占节点层总耗时 |
| --- | ---: |
| retrieve | 42.7% |
| route | 25.3% |
| generate | 18.5% |
| tools | 6.8% |
| prepare | 6.7% |
| clarify | 0.0% |

调用层中耗时最多的 LLM 用途（生成/改写/路由）与检索步骤见 5.1；节点层与调用层是包含关系，不做加总。

