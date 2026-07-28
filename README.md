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

测试不访问外部 LLM 服务。测试同时覆盖独立 blind fixture、同义改写、严格 schema、LLM 脱敏/兼容降级、Hybrid 分歧、并发快照及 CSV 安全。

## 检测分类

本项目将以下情况定义为幻觉：客服回复中的事实、参数、政策或办理方式与知识库明确冲突；回复声称完成系统不具备的查询或操作；回复给出知识库没有依据的具体事实；回复忽略关键限制，导致用户得到实质性错误结论。仅仅没有复述知识库的全部内容不算幻觉，除非遗漏会改变用户决策或安全判断。

严重程度分为四级：`严重` 表示可能造成健康或人身安全风险；`高` 表示可能影响资金、消费者权益、交易或关键操作；`中` 表示会影响购买决策或服务体验但通常可纠正；`低` 表示影响有限的非关键偏差。正常回复标记为 `无`。具体 case 可根据影响范围在默认级别上调整。

| 分类 | 定义 | 默认严重程度 | 选择理由 |
| --- | --- | --- | --- |
| 政策与优惠 | 退换货、发票、优惠、发货时效或办理流程与知识库不一致 | 高 | 容易造成资金损失、权益争议或错误操作；仅一般物流时效偏差可降为中 |
| 产品事实与参数 | 产品材质、接口、版本、功能、保修等信息错误或无依据 | 高 | 会直接影响购买决策、兼容性和售后预期 |
| 能力越界 | 声称已查询物流、修改订单、升级工单等，但系统实际不具备该能力 | 高 | 会让用户误以为关键操作已经完成，可能延误后续处理 |
| 无依据事实 | 编造地址、门店、品牌关系等知识库未支持的具体事实 | 中 | 通常影响信任和决策；错误退货地址等可能造成财物损失的 case 升为高 |
| 安全误导 | 对健康、成分或使用风险作出与警示信息相反的确定性保证 | 严重 | 可能直接造成健康或人身安全风险 |
| 关键信息遗漏 | 遗漏足以改变结论的重要限制、比例或建议，并给出过度确定的回答 | 中 | 难点不是普通省略，而是遗漏后结论已具有误导性 |

离线引擎除领域规则外，还比较产地、颜色、期限等事实槽位，并对“一年/十二个月”等等价值做归一化。

## 检出率与评估结果

检测阶段只读取 `data/replies.json`，完成 20 条逐条标注后，评估阶段才加载 `data/ground_truth.json`。最终离线评测结果如下，详细数据见 `outputs/metrics.json` 和 `outputs/predictions.json`。

| 指标 | 结果 |
| --- | ---: |
| 样本总数 | 20 |
| TP / TN / FP / FN | 18 / 2 / 0 / 0 |
| Precision | 1.0 |
| Recall（检出率） | 1.0 |
| F1 | 1.0 |
| Accuracy | 1.0 |
| 漏检 case | 无 |
| 误报 case | 无 |

## 易误判 case 与局限性

当前固定开发集没有产生漏检或误报，但以下 case 在新数据或改写表达中容易误判：

- `h09`（NFC 功能）：知识库“未标注某功能”不一定等于产品明确“不支持”。当前业务口径要求客服不能把未知信息肯定为支持，但若知识库本身不完整，可能造成误报。改进方式是区分“明确否定”“未知”和“有外部证据支持”三种状态。
- `h20`（鞋码建议）：这是关键信息遗漏而非直接事实冲突，需要判断被遗漏内容是否足以改变结论。规则可能漏掉更隐含的限定条件。可通过句级主张抽取、关键信息权重和 LLM 复核改进。
- `h12`、`h16`（正常回复）：客服回复没有逐字复述知识库，但核心结论一致。过度依赖关键词或要求完全覆盖知识库，容易把合理概括误报为幻觉。应使用语义一致性判断并区分“非穷举回答”和“误导性遗漏”。
- `h03`、`h10`、`h14`、`h18`（能力越界）：检测准确性依赖知识库是否及时描述系统能力。如果能力清单缺失或已过期，可能误判。生产环境应从权限与工具注册表动态读取能力，而不是只依赖自然语言说明。
- 数字、单位和多事实回复：同一句话可能同时包含正确与错误内容，且“一年/12个月”等表达需要归一化。当前事实槽位词表仍有限，可通过更通用的实体属性抽取、单位换算和独立盲测集持续扩充。

标准 20 条开发集满分只说明对该固定集合无误报漏报，不代表生产泛化能力。离线规则仍可能无法处理未知属性、复杂否定、隐含关系或全新语言表达；上线前必须使用独立、持续扩充且不参与规则开发的评测集验证。

## AI 工具使用情况

项目使用 Codex 多会话协作完成：规划会话负责方案与任务拆分，执行会话负责实现，检查会话独立审查安全性和评估正确性，验收会话负责真实浏览器与移动端验证。AI 用于辅助编码、测试、代码审查和界面验收；最终结果均通过自动化测试与独立复核。检测阶段不会读取人工标签，避免将评测答案泄漏到检测逻辑。
