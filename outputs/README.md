# 采集与测量材料

成果入口见 [RESULTS](../docs/RESULTS.md)。阶段交付、验收与返工报告保留本地，固定历史链接由成果文档统一提供。

本目录保留：

- `live_baseline_20261011/run_capture.py`：基线采集与首错停止逻辑，被测试和优化后包装器引用。
- `live_opt1_20261011/run_capture_opt1.py`：显式使用优化后冻结场景的采集入口。
- 两个 `pricing.json`：2026-10-11 冻结价格，不代表当前报价或实际账单。
- `measurement_report.md`：历史基线报告副本，不能替代原始记录。
- `verify_capture.py`、`protected_before.json` 和三个检查 JSON：历史核验脚本、必要输入及结果。

核验脚本需要原开发机 `logs/performance/` 中的原始记录及相应保护文件，不是全新克隆即可复算的独立数据包；本机专用路径不可当成通用安装路径。真实采集可能产生外部服务费用，普通回归无需运行采集入口。
