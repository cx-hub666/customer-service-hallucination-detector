# 客服回复幻觉检测

一个将检测和评估严格隔离的客服回复事实核查工具。它提供确定性的离线规则、OpenAI-compatible LLM 模式、混合模式、命令行接口和 Flask 仪表盘。

## 数据与输出

- `data/replies.json`：检测输入，仅含 `id`、`user_question`、`system_reply`、`knowledge_base`。
- `data/ground_truth.json`：仅在评估阶段读取的人工标签。
- `outputs/predictions.json`：逐条检测结果，包含原始 `user_question`。
- `outputs/predictions.csv`：便于审阅的 CSV 结果。
- `outputs/metrics.json`：混淆矩阵、Precision、Recall、F1、Accuracy 和分类分布。

检测输入中任意层级出现 `ground_truth`、`is_hallucination`、`label`、`expected` 等评估字段都会被拒绝。检测器不读取 `ground_truth.json`，规则也不依赖样本 ID。评估阶段要求预测和人工标签显式包含严格的 boolean `is_hallucination`；缺失字段、字符串布尔值、非法分类或不完整预测 schema 会返回明确错误。

## 环境

需要 Python 3.10 或更高版本。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

离线模式无需 API Key。LLM 和混合模式可参考 `.env.example` 配置 `LLM_API_BASE`、`LLM_API_KEY` 和 `LLM_MODEL`；程序不会自动读取 `.env`，请通过进程环境或部署平台注入。

上游 LLM 错误正文不会返回给 Web 客户端，也不会由本项目记录原始 `Authorization`。即使上游回显请求头，客户端只会收到固定的脱敏错误。`response_format` 不兼容时允许一次独立的纯 JSON prompt 降级，该次兼容请求不占用常规重试预算。

## CLI

三条标准命令如下：

```powershell
python -m hallucination_detector detect --input data/replies.json --output outputs/predictions.json --mode offline
python -m hallucination_detector evaluate --predictions outputs/predictions.json --ground-truth data/ground_truth.json
python -m hallucination_detector run --input data/replies.json --ground-truth data/ground_truth.json --output-dir outputs --mode offline
```

`detect` 只检测；`evaluate` 只评估已有预测；`run` 先完成检测并落盘，再单独加载标签进行评估。

## Web 仪表盘

```powershell
python app.py
```

打开 [http://127.0.0.1:5000](http://127.0.0.1:5000)。仪表盘支持运行三种模式、查看指标和混淆矩阵、筛选检测明细，以及导出 JSON/CSV。
<img width="865" height="490" alt="image" src="https://github.com/user-attachments/assets/b4b2bcba-5496-40f4-96d1-75f2eb18b705" />
<img width="865" height="506" alt="image" src="https://github.com/user-attachments/assets/a47e64db-04ba-404b-a7f6-1d9e51d1bed1" />

主要接口：

- `POST /api/detect`：请求体 `{"mode":"offline"}`，只执行检测并使旧指标失效。
- `POST /api/evaluate`：评估最近一次预测。
- `POST /api/run`：检测并评估。
- `GET /api/results`：读取最近结果。
- `GET /api/export/json`、`GET /api/export/csv`：下载预测。

`detect` 和 `run` 要求 `Content-Type: application/json`、合法 JSON 对象及显式 `mode`。非 JSON、畸形 JSON、数组或缺失模式均返回结构化 400。Web 运行通过进程内锁和 `run_id` manifest 发布原子快照，`results` 与导出只读取同一次运行的数据。CSV 使用 UTF-8 BOM，并对以 `= + - @` 开头的文本添加安全前缀以避免电子表格公式注入。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试不访问外部 LLM 服务。测试同时覆盖独立 blind fixture、同义改写、严格 schema、LLM 脱敏/兼容降级、Hybrid 分歧、并发快照及 CSV 安全。当前标准数据的离线目标为 `TP=18, TN=2, FP=0, FN=0`。

## 检测分类

输出使用六个统一分类：`政策与优惠`、`产品事实与参数`、`能力越界`、`无依据事实`、`安全误导`、`关键信息遗漏`。离线引擎除领域规则外，还比较产地、颜色、期限等事实槽位，并对“一年/十二个月”等等价值做归一化。

标准 20 条开发集满分只说明对该固定集合无误报漏报，不代表生产泛化能力。离线规则仍可能无法处理未知属性、复杂否定、隐含关系或全新语言表达；上线前必须使用独立、持续扩充且不参与规则开发的评测集验证。
