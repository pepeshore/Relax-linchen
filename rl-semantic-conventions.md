# 大模型在线 RL 语义规范草案

**Status:** Development

**Version:** 0.21.0

**Date:** 2026-11-11

本规范定义大模型在线 RL 的阶段 span 和 Agent 调用 span，覆盖其含义、名称、起止边界、状态、采集要求及 attributes，用于阶段耗时分析和轨迹下钻。

本文采用 [OTel GenAI client inference](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md) 的 span 定义格式。`rl.*` 是拟议的通用命名空间，尚不是 OTel 官方语义约定；Relax、verl 的组件名称不限定本规范的逻辑职责。SGLang、Megatron 内部埋点直接复用，不另行定义。

## 1 对象与约定

| 对象              | 定义                                                                                                |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| Run               | 一次实际训练运行；重新启动作业形成新运行身份                                                        |
| Iteration         | 框架外层 RL 轮次，不是 epoch、minibatch、microbatch 或优化器更新次数                                |
| Step              | 一个有界处理单元；适配器说明其计数器归属和完成条件                                                  |
| Batch             | 某操作处理的输入集合；过滤、拆分或合并改变成员集合时产生新 ID                                       |
| Group             | 针对同一输入的一次成组采样；同一题再次采样是新的 Group                                              |
| Sample            | 一条输入题目的一次完整采样调用；暂停恢复保持身份，重复采样使用新身份；可产出零个、一个或多个 Export |
| Session           | 一次 Agent 调用的身份；暂停恢复保持不变，重新启动调用产生新身份                                     |
| Request / Attempt | 一次逻辑模型请求，以及其中一次实际后端调用／恢复片段                                                |
| Export            | Session 导出的一个训练轨迹对象；一个 Session 可以有多个 Export                                      |
| Update            | 一次优化器更新决策，可提交、跳过或失败；一次 train 操作可以有多个 Update                            |

Role 表示逻辑职责，如 actor、rollout、reference；它不是所有框架已有的统一组件名，也不是 Relax 专属。同一进程可以承担多种职责。

`rl.iteration.index` 保留框架原始外层轮次编号，适配器说明从 0 还是 1 开始。只有确认框架 Step 符合该含义时才能直接映射。Run、Step 是业务关联上下文，按第 2 章聚合，不新增统一父 span。

所有新增 `rl.*` 属性的 Stability 为 Development。Required 表示必须提供；Conditionally Required 表示表中条件成立时必须提供。未知信息省略，不用 0、false 或空集合代替未知。ID 使用 string，除全局唯一 Run ID 外，至少在 Run 内唯一。

## 2 通用要求

阶段 span 由掌握操作边界的一方记录一次，保留原生起止，包括该调用内部的通信、等待和重试。提交成功不代表 GPU 工作完成；未执行的阶段不生成占位 span。训练请求和各 rank 分别记录本地时间，不能将不同 rank 的最早开始与最晚结束拼成一次实际调用，也不能累加 rank 耗时。

### 计时口径

Relax 记录 host wall-clock 调用时间，不是 CUDA kernel 时间。异步 GPU 工作可能在 host 函数返回后继续执行；不得为 trace 新增 barrier、CUDA synchronize 或调整已有训练调度。同一 rank 的串行调用应保留实际先后顺序，不根据图形裁剪或平移 span。

| Key                                         | Stability   | Requirement Level              | Value Type | Description                                                                                                      |
| ------------------------------------------- | ----------- | ------------------------------ | ---------- | ---------------------------------------------------------------------------------------------------------------- |
| `rl.timing.clock`                           | Development | Required                       | string     | 当前 RL 实现为 `host_wall`，不表示 GPU 时间                                                                      |
| `rl.timing.scope`                           | Development | Required                       | string     | `operation` 为一次调用，`rank_local` 为 Megatron worker 本地调用，`source_sync` 为发送方观察到的权重交付或激活。 |
| `rl.rank.global`                            | Development | Required：rank 及其训练阶段    | int        | Ray 训练组分配的全局 rank。                                                                                      |
| `rl.rank.world_size`                        | Development | Required：训练请求和 rank 入口 | int        | 该请求派发的 rank 数或 worker 所在组的规模。                                                                     |
| `rl.timing.rank.dp` / `.cp` / `.tp` / `.pp` | Development | Conditionally Required         | int        | 进入实际计算时从可用的 Megatron 并行组读取的本地坐标。                                                           |

Relax 在 Ray 训练组分发一次 Megatron 请求时创建一个 `rl.actor.train_step` 或 `rl.critic.train_step`。起点为分发开始，终点为发送端观察到所有已派发 RPC 完成；同步提交失败时结束并记录错误。这个 span 使用框架进程的时钟。每个收到请求的进程在 RPC 入口创建 `rl.actor.train_rank` 或 `rl.critic.train_rank`，以请求的 W3C traceparent 为直接 parent，使用本进程时钟计时、落盘。rank 的 reference／teacher／actor／critic forward、advantage、actor／critic train 直接挂在本 rank 入口下。每个阶段只记录实际执行时间、本地成员和本地可获取的结果；CP 的 token 数和均值是本分片数据。无法确认成员时省略 batch.size 和相应成员明细。0.13–0.17 历史数据使用代表 rank 和 `rl.timing.ranks`。

advantage 只采集真正执行计算的末级 PP，其他 PP 的直接返回不是计算。异步独立 forward 仍按实际 DP 分区调用记录，属于 `operation`，不能冒充全局批次耗时。无 rank 汇总的 rollout、reward、Agent、过滤、数据传输和 CPU advantage 记录各自实际生命周期。跨主机调用比较需要时钟同步；图中的重叠本身不证明 GPU 并行。

操作返回或失败时先固定结束时间，再整理 observation JSON 和导出；观测结果的序列化、异步统计读取、checkpoint 或导出完成时间不得延长业务 span。批量评分若有真实的批次后处理，即使返回值未改变也必须计入；内建纯逐条 fanout 结束于最后一个真实评分调用。可恢复的 ABORTED Sample 不冻结终态时间，恢复后沿用首次起点。

权重 span 的 `source_sync` 是源端同步操作观测：publish 从同步准备到已有交付确认，load 从同次同步准备到目标确认可用及恢复完成。两者可真实重叠，包含暂停、缓存清理、转换、通信及确认等待；LoRA 发布流程还可能包含恢复生成。当前没有目标侧独立安装起点，因此这些 spans 不表示纯网络传输或 GPU 安装耗时，不能把源端起点解释成目标开始安装。

span 本身的成功／失败统一使用原生 Span Status，不另外定义同义 attributes。本文的 `status_code` 和 `status_message` 分别指 OTLP 的 `span.status.code` 和 `span.status.message`；SDK 通常将 message 称为 description。

| 情况                                           | status_code | status_message                                                  |
| ---------------------------------------------- | ----------- | --------------------------------------------------------------- |
| 操作正常结束，未发生未处理错误                 | `UNSET`     | 不设置                                                          |
| 操作最终失败，如超时、异常退出或未完成约定交付 | `ERROR`     | 简短说明实际失败原因，如 `Timed out waiting for reward service` |
| 应用显式确认操作成功                           | `OK`        | 不设置；本规范的埋点默认不主动设置 `OK`                         |

`status_message` 仅用于 `ERROR`；`UNSET`／`OK` 的 message 会被忽略，不能用它保存正常业务结果。运行中 span 也可能是 `UNSET`，因此不能只根据状态码判断已完成。以上规则遵循 [OTel Set Status](https://opentelemetry.io/docs/specs/otel/trace/api/#set-status)。

正常零 reward、长度截断、过滤 drop、按策略跳过更新，以及约定允许的取消或部分交付，不自动构成错误。若未完成该操作承诺，则设置 `ERROR`；重试后已经恢复的错误不把整个操作标为失败。逐 Export 的过滤决定、逐 Update 的 committed／skipped／failed、逐 Attempt 的结果等保留在对应阶段的业务明细中；这些记录描述各个对象，不重复表示整个 span 的成功／失败。

本版不额外要求 `rl.outcome`、`rl.reason` 或错误分类属性。OTel [错误记录建议](https://opentelemetry.io/docs/specs/semconv/general/recording-errors/#recording-errors-on-spans)中的 `error.type` 用于错误分类；已有埋点提供时可原样保留，不作为页面判断操作失败的必要条件。

span 名称固定，不拼接 Run、Step、rank、模型名或请求内容。角色、动作由名称表达，不重复增加同义属性。`service.name`、`service.instance.id` 和主机／进程身份放在 Resource；埋点库 name/version 放在 Instrumentation Scope。采集器保留真实来源身份。

属性按事实所属的阶段记录，不向后续 span 复制整套上下文：

| 属性                             | 记录位置                                                                                                                                | 口径                                                                                               |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `rl.batch.id`、`rl.batch.size`   | 所有批次操作 span                                                                                                                       | 各阶段记录自身实际输入集合身份与成员数；真实子操作 span 只记录所属 batch.id，不重复记录 batch.size |
| `rl.producer.role`               | Development                                                                                                                             | Required                                                                                           |
| `rl.execution.side`              | 本次 rollout 拥有的评分批次、动态过滤、队列写入、自定义 advantage、OPD 前向；训练侧的前向、优势及训练请求、rank 入口和各 rank 训练 span | string：`rollout` 或 `training`，表示实际执行侧。独立调用且执行侧未知时省略。                      |
| `rl.purpose`                     | rollout、样本、Agent、评分及已确定归属的 rollout 子阶段                                                                                 | `training` 或 `evaluation`，描述采样用途                                                           |
| `rl.algorithm.name`              | `rl.actor.train`                                                                                                                        | 记录本轮策略训练算法；优势阶段单独记录实际 estimator，Critic 不重复复制策略算法名                  |
| 评分、优势、过滤、传输和训练结果 | 各自执行该操作的 span                                                                                                                   | 保留逐 Export／Update 结果，后续按对象 ID 关联，不复制到 Agent span                                |

生成批次以输入任务为成员，训练批次以实际接纳的训练样本为成员，不能直接沿用生成批次的 ID 或数量。Actor 和 Critic 分别记录自身实际接纳的批次；仅在样本身份和成员集合完全一致时才可使用相同批次 ID。批次规模描述本阶段工作量，不能将各阶段的 size 相加作为全局样本总数。

Run／Step／Iteration 是查询和关联所需的上下文，仍按各表条件记录；模型、后端和版本描述本阶段实际执行对象，也可能因阶段不同而不同。各 span 的属性表列出其完整属性要求；未列出的其他阶段属性不复制，parent 或 Link 不会自动继承属性。轨迹流转使用逐 Export ID 关联。

**Trace 与关联规则：** 同一 iteration 的核心阶段在条件允许时复用 traceId，但不强制共用 trace；一次框架 rollout 直接拥有样本、评分批次以及它发起的动态过滤、队列交付、Agentic 自定义优势计算和 OPD 推理前向；同一阶段的批次与真实独立成员操作按第 2.1 节建立父子关系，结果数组中的记录不构成子 span。通过业务 ID 关联查询，通过 Span Link 表达实际依赖。

1. **Trace 复用。** 第 3 章同一 iteration 内的核心阶段，已有合适的共同上游 Context 时应复用同一个 traceId，各次操作使用不同的 spanId，有明确包含关系时使用实际调用者作为 parent（例如 rollout 内的评分批次），其他阶段作为共同上游下的同级 span。没有这样的 Context 时，各批次阶段可以创建独立 trace 的根 span；本规范不为复用 traceId 新增统一的 Run／Step 父 span。按照 [OTel Span Creation](https://opentelemetry.io/docs/specs/otel/trace/api/#span-creation)，同一 trace 中的 spans 应属于同一棵调用树，不能将多个无 parent 的根 span 手动设置为相同 traceId。第 4 章的 Agent 调用是对应 sample 的子 span，暂停恢复沿用原 span 和 traceId。
2. **业务聚合。** 无论是否复用 traceId，都通过 `rl.run.id` 聚合运行，通过 `(rl.run.id, rl.step.id)` 关联查询同一 Step 的各阶段和轨迹；查询必须支持一个 Step 对应多条 trace，不能依赖 traceId 完成整个 Step 的聚合。明确唯一 Step 归属时必须填写 `rl.step.id`；未知、未分配或跨 Step 时省略单值，按已有的真实业务归属明细关联，不能任选一个 Step。相同 Run／Step 只表示业务归属，不自动构成依赖关系。
3. **依赖关联。** 无论同 trace 还是跨 trace，有实际调用触发、数据消费或参数版本来源关系时，由被触发方／消费方的 span Link 到直接来源 span 的真实 SpanContext；parent 表达调用归属，Link 表达数据或版本来源。例如，训练 Link 到实际消费的优势计算／数据交付，权重加载 Link 到对应权重发布。多来源操作关联各实际直接来源，并按 `(traceId, spanId)` 去重；不因同轮、时间相近或仅先后执行而建立 Link，不要求补齐所有间接祖先。
4. **上下文传递。** 跨进程、线程、队列及权重交付时，按实际关系传递 Context：符合第 1 条的共同上游 Context 可用于创建同级阶段、复用 traceId；rollout 内发起的阶段在实际开始时使用所属 rollout 的 Context 作为 parent，样本或其他数据来源的 Context 用于 Link。创建时已知的依赖直接加入 Links，执行中确认的依赖在 span 结束前追加；不修改已经结束的来源 span。来源未被采集或上下文确实不可取得时，保留已有业务 ID 和明细，不根据 Run／Step／Export ID 生成虚假的 traceId 或 spanId。
5. **后端工作的 RL 封装。** SGLang、Megatron 的后端工作必须归属于负责该次操作的 RL 框架 span，由该 span 记录框架侧调用的完整起止与结果。直接复用本规范已有的阶段 span，不为每个后端内部 span 另建 RL 包装 span。例如，策略训练归属于 `rl.actor.train`，价值训练归属于 `rl.critic.train`，独立前向归属于对应的 forward span；普通生成归属于 `rl.rollout.sample`，其 parent 为首次接纳的 `rl.rollout.generate`；Agent 调用中的生成归属于 `rl.agent.run`，其 parent 是对应 sample。
6. **SGLang 推理父子关系。** SGLang 请求入口 span 继承负责本次调用的 RL span 的 traceId，并将其作为 parent：普通生成指向 `rl.rollout.sample`，Agentic 模型请求使用第 2 章下述回调规则。通过 HTTP `traceparent`／`tracestate` 传递真实 SpanContext，在进入 Ray HTTP worker 或 router 前完成注入；中间层应保留该上下文。SGLang 内部 spans 保持原有名称和父子树，不为每个内部 span 再创建 RL 包装。单个 sample／Agent 可以包含多次请求，它们可以经不同的 Agent 应用 span 归属于同一个 RL span；重试和恢复不把请求挂到其他样本。若启用 router 自身 tracing，允许真实的 router span 位于 RL 与 SGLang 之间，并保持可追溯的父链。
7. **其他后端与耗时。** 其他后端（如 Megatron）仍可使用独立 trace，通过入口 Span Link 指向负责该操作的 RL span；Relax 的 Megatron 训练 RPC 通过 W3C 上下文关联各 rank 和本地阶段；外部 Megatron profiler 事件仍按其自身采集边界解释。SGLang 与 RL 共享 trace 后，一条 rollout 批次 trace 可以包含多个样本及其推理细节，不再要求这些 SGLang 请求使用独立 trace。各阶段保留真实起止；父子、并行阶段及 RL／后端耗时不能相加作为 Step 总耗时。

**当前实现范围（Relax RL 0.21.0）：** 内置普通 rollout、`GenerateState.post_generate` 和 Agentic SGLang adapter 注入上述 W3C 上下文。RL tracing 关闭或无有效来源 span 时不注入。现有路由 headers 保留；独立并发样本／Session 的上下文相互隔离。自定义生成若绕过这些入口、自行调用 HTTP 客户端，需使用相同的 `inference_headers` 辅助函数。

SGLang 仍需启用其 tracing 并配置 OTLP 导出，后端 spans 不会由 RL 的进程内文件 exporter 自动写出。采集两端后按 `(traceId, spanId, parentSpanId)` 还原父子关系；`rl.run.id`／`rl.step.id` 等属性不会由 parent 自动复制，可沿父链定位所属 RL span。不要用时间接近或业务 ID 伪造父关系。

**Agent 应用上下文：** Session 启动子进程时提供 `RELAX_TRACEPARENT` 和可选的 `RELAX_TRACESTATE`，值来自该 Session 的 `rl.agent.run`。应用提取此 carrier 后启动自己的 span，回调时注入当前 span。Relax 仅保存这次逻辑请求的 W3C headers，重试和恢复沿用。有效且 traceId 与此次采样一致时，发送给 SGLang 的父上下文采用来路的 parent-id、flags 和配套 tracestate。缺失、无效或跨 trace 时重新从当前 Session 的 `rl.agent.run` 注入，丢弃来路 tracestate。同一 rollout 的多个 Session 共享 traceId；应用负责隔离各 Session 的上下文。Agent 自身的 span 由应用 SDK 导出。

示例 Agent 在应用 scope `relax.example.agent` 中为每次收到的任务创建 `agent.invoke`，作为 `rl.agent.run` 的子 span。应用中的实际 LLM 请求与工具执行分别使用 `llm.call` 和 `tool.call`，继承当时的活动父 span。发往 Relax 的推理回调携带该次 `llm.call` 的 W3C 上下文。工具执行 span 从调用开始，到同步或异步结果返回或抛错时结束。模型仅提出工具调用、应用未执行时，省略工具 span。

结构化数据直接保存为 span 的 string 类型 JSON 属性，包括题目、response、评分、优势、过滤、传输及训练结果。自 0.10.0 起，新采集不再输出对应的 `_ref`；完整内容随 OTLP JSONL 保存，不依赖附件或旧详情接口。禁止按长度截断，也不使用随机采样结果冒充全量数据。读取工具可以兼容旧版 `_ref`，但不能将旧引用当作新数据已内联的证据。

结构化内容中的 ID 为 string，计数为非负 int，结果数值为 double，布尔值为 boolean，`start_time`／`end_time` 为 Unix 秒 double。未知时间边界省略，后端原始字段保留原单位。普通 SDK 通常在 span 结束后导出；运行中展示需要同一 span 身份的状态／数据快照，终态记录去重合并，不增加业务 span 类型。

## 2.1 批次、独立样本操作与结果明细

每个实际批次操作生成一个总 span，记录该操作的 `rl.batch.id`、实际输入数 `rl.batch.size` 和完整结果。只有存在真实、可独立观测起止和结果的成员操作时才创建样本级子 span。结果数组、逐样本统计或样本参与记录本身不构成独立操作；这些明细直接保留在批次的 JSON attributes 中，不生成零时长 span 或虚构时间事件。

| 实际操作                     | 总 span                                                                               | 子 span                | 结果明细位置与判断                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------- |
| 普通／Agentic 完整 rollout   | `rl.rollout.generate`                                                                 | 样本及实际发起的子阶段 | 样本和评分的归属见下文；过滤、交付、自定义优势和 OPD 前向也按实际调用挂接                       |
| 逐条调用评分器               | `rl.reward.batch`                                                                     | `rl.reward.score`      | 保留：每条 Export 有独立评分调用，父级保留全批结果，子级保留自身结果                            |
| 一次调用批量评分器           | `rl.reward.batch`                                                                     | 无                     | `rl.reward.results` 保存所有成员；不能根据返回数组补造 score span                               |
| Megatron 批量前向            | `rl.reference.forward`、`rl.teacher.forward`、`rl.critic.forward`、`rl.actor.forward` | 无                     | `rl.forward.results` 保存成员结果；packed tensor 执行没有单样本调用边界                         |
| OPD 单样本前向               | `rl.teacher.forward`、`rl.actor.forward`                                              | 无                     | 一次真实调用只记录一个 span，batch.size=1，直接带样本身份与结果，不重复包装相同调用             |
| 优势计算（包含自定义组计算） | `rl.advantage.compute`                                                                | 无                     | `rl.advantage.results` 保存逐成员统计；优势可依赖整个 Group，统计行不是独立计算                 |
| 过滤                         | `rl.data.filter`                                                                      | 无                     | `rl.filter.results` 保存逐成员 keep/drop；当前组／批次决策不拆成虚构样本操作                    |
| 数据传输                     | `rl.data.transfer`                                                                    | 无                     | `rl.transfer.data` 保存成员与实际发送／ACK；共享批次 ACK 不拆成独立交付耗时                     |
| Actor／Critic 训练           | `rl.actor.train`、`rl.critic.train`                                                   | 无                     | `rl.train.results.uses` 保存样本参与记录，`updates` 保存优化器更新；训练按批次／microbatch 执行 |

生成批次按输入 Sample 计数，评分等下游批次按实际输入 Export 计数，不能假定两者总数相同。真实子操作与总 span 共用 traceId，parentSpanId 指向真实总 span。样本与 rollout 发起的各阶段都是生成 span 的直接子操作。评分、过滤、交付、优势与前向各自保留本阶段的批次身份和计数。逐成员的数据来源通过 Link 关联：例如逐条 `rl.reward.score` 的 parent 是 `rl.reward.batch`，其 Link 指向实际消费的 `rl.rollout.sample`。批量评分没有 score 子 span 时，后续阶段 Link 到实际 `rl.reward.batch`；过滤与传输同样发布批次来源，保留逐成员业务 ID。SDK 显式使用真实父 Context，不手动拼接 traceId。

**完整调用与窗口：** Sample 暂停／可续跑中断不结束 span，恢复沿用同一 sample.id、spanId、traceId 和首次 parent；后续窗口通过 Link 和 `rl.sample.memberships` 记录实际归属，不重新挂 parent。一个完整 Sample 可以晚于首次接纳窗口结束。框架每次调用 rollout 函数（训练和评估分别计）创建一个生成 span，跨 Group 和补交共享。预取的 Session 继承发起准备的调用上下文，首次接纳后保留原 parent，实际消费轮次记在 sample 的 memberships 中。若调用返回时预取仍在决定是否接纳 Session，生成 span 可延后导出以记录准确的批次大小，结束时间仍取调用返回或抛错的时刻。

**时间真实性：**

- 样本级子 span 使用 `rl.sample.timing=operation`，起止取自真实成员调用；不复制或均分批次耗时。父、子及并行 span 的耗时不能相加作为阶段耗时。
- 本版不生成 `rl.sample.timing=result` 的零时长结果确认 span，也不根据结果数组补造 `.sample` 子 span。汇总、异步导出或 checkpoint 完成时间不是样本计算时间；延迟导出仍保留批次原生起止。
- 旧版 0.10.0 的 result span 可兼容读取，但查看器应与执行时间线分开，默认展示批次上的结果明细。不得修改已有原始 trace 或将重绘结果宣称为重新训练的数据。

真实子操作为 `INTERNAL`，遵循第 2 章状态规则。正常零分、drop、skipped 仍为 `UNSET`；未启动的成员不伪造执行 span，已启动后失败的操作应保留。

**公共属性（补充各阶段属性表）：**

| Key                     | Stability   | Requirement Level                                                 | Value Type | Description                                                |
| ----------------------- | ----------- | ----------------------------------------------------------------- | ---------- | ---------------------------------------------------------- |
| `rl.batch.id`           | Development | Required：批次 span 及真实子操作 span                             | string     | 当前阶段的实际输入集合身份；子操作填真实 parent 的批次身份 |
| `rl.batch.size`         | Development | Required：批次输入确定时                                          | int        | 实际输入成员数，不按 rank 重复，不记在子操作 span 上       |
| `rl.prompt.id`          | Development | Conditionally Required：rollout.sample／reward.score 有题目来源时 | string     | 原始题目身份，同一题目多次采样共用                         |
| `rl.sample.id`          | Development | Required：rollout.sample；其他子操作可取得时                      | string     | 一次输入采样调用身份；同题重复采样不同，续跑保持不变       |
| `rl.export.id`          | Development | Required：下游成员使用 Export 身份时                              | string     | 该结果所属的真实 Export ID，不以样本下标冒充               |
| `rl.sample.index`       | Development | Conditionally Required：框架只传递样本下标时                      | string     | 框架实际输入样本下标，与 run/step 一起关联；不是 Export ID |
| `rl.sample.timing`      | Development | Required：真实子操作 span                                         | string     | 仅 operation，表示真实成员调用                             |
| `rl.sample.memberships` | Development | Conditionally Required：完整调用涉及多个窗口／Step                | string     | JSON 数组，保存实际 batch_id、step_id，不覆盖历史          |

异步独立前向可按实际 DP 分区调用生成独立批次，`rl.partition.dp_rank` 记录整数分区编号；每个批次只计本分区真实成员，TP/PP/CP 不重复创建同一成员。训练 RPC 内可通过已有返回路径合并各 rank 的事实。

前向的 `rl.forward.results` 为 JSON 数组，保存真实 `export_id` 或 `sample_index` 及结果状态；未独立采集的 token 概率不伪造。训练的 `rl.train.results.uses` 保存每条实际消费，通过 update_id 关联同一对象内 updates 的 committed／skipped／failed 状态。传输的 `rl.transfer.data.members` 可补充逐成员 export_id、sample_index；共同发送／ACK 时间只记录在批次上。

下游批次及真实子操作 span 保留 `rl.run.id`、可确定的 `rl.step.id`／`rl.iteration.index` 及该阶段必要属性。JSON 结果记录通过所在 span 取得运行上下文，成员使用真实 `export_id` 或 `sample_index` 关联。普通训练仅有 sample_indices 时，span 使用 `rl.sample.index`，结果记录使用 `sample_index`；rollout／评分同时保留同一个实际 index，不能凭空生成 Export 映射。数据不存在时不生成占位结果，采集不完整需明确诊断。

### 问答：怎样按题目找出所有回答和分数？

在同一 `rl.run.id` 内，按 `rl.prompt.id` 找到这道题的全部 `rl.rollout.sample`。同题的每次采样使用不同的 `rl.sample.id`。从样本 span 的 `rl.sample.response` 取回答，按样本 ID 关联 `rl.reward.score` 或 `rl.reward.batch` 的 `rl.reward.results[]`。一次采样产生多份 Export 时，再按回答和评分结果中的 `export_id` 匹配。即使同一批含多道题，也按每条结果的 `prompt_id` 分组；只调用批量评分器时可直接查这些结果行。Span Link 继续表达数据来源。

内置 JSONL／Parquet 数据集在套用模板、打乱和重复采样前，复用 Agentic 数据清单的原始行标识。标识为 `prompt_` 加来源快照的 SHA-256，快照包括来源文件绝对路径、原始行号、数据集／版本、split 和原始行内容。同一来源快照在普通与 Agentic rollout、eager 与 streaming 加载及多次抽样中一致。相同文本位于不同行时各自有 ID。文件迁移、split 或原始内容变更可能产生新 ID。

自定义数据源可提供 `Sample.prompt_id`，也可沿用 `Sample.agentic_prompt_source["prompt_id"]` 或 `Sample.metadata["prompt_id"]`，按此顺序取首个有效值。字符串和整数形式的 ID 统一输出为字符串。对缺少稳定题目标识的自定义输入或旧数据，省略题目 ID，仍可按已有的样本、Export 身份查询。

### 问答：怎样区分训练侧和 rollout 侧的同名 span？

查 `rl.execution.side`。在普通和 Agentic rollout 中，实际发起的动态过滤 `rl.data.filter`、队列交付 `rl.data.transfer`、Agentic 自定义 `rl.advantage.compute`、OPD 的 `rl.teacher.forward` 和 `rl.actor.forward` 都标为 `rollout`。每个操作与当次 `rl.rollout.generate` 共享 traceId，parentSpanId 指向它。新根按实际框架调用划分；一次 step 可以有多次调用。相同 run、step 或样本来源 Link 单独都不足以确定 parent。单独调用专用的 OPD prefill、Agentic 自定义优势计算或 rollout 队列写入时，仍可从入口确定执行侧为 `rollout`，但保留根 span。独立调用通用过滤器或评分器、执行侧未知时省略此属性。

Agentic 常驻任务在操作开始时固定 parent 和 Step。上一轮启动并跨轮完成的操作保留原归属。恢复后新启动的操作归当时的 rollout。在两轮之间独立开始、没有活动调用的操作省略 parent 和未知 Step。写队列回填旧分区时，`rl.step.id` 属于发起本次写入的 rollout，`rl.transfer.target.step_id` 表示实际写入的目标。rollout 函数返回后调用的样本过滤回调不在该调用范围内。

训练侧 Megatron 的前向和优势计算、独立 Advantages 服务均标为 `training`，保留各自的调用边界与原生计时。例如，训练侧的 `rl.actor.forward` 可以是 `old_policy` 或 `distillation`；rollout 侧的 OPD student prefill 使用 `distillation`。按 `rl.execution.side` 区分谁执行，按 `rl.forward.purpose` 区分计算用途。一次训练采样中的 rollout 子操作可以同时具有 `rl.execution.side=rollout` 和 `rl.purpose=training`。

### 问答：如何判断 span 由哪个角色创建？

从 0.19 起读取 span 的 `rl.producer.role`。同一 trace 中，各进程写自己的角色。主机、PID 和 `rl.rank.global` 可进一步定位实例。

| 角色值                            | 创建方与示例                                                                                                                                       |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rollout_manager`                 | RolloutManager 发起整轮生成，在本进程执行的普通采样、评分、过滤和传输也使用此值。                                                                  |
| `agentic_session_shard`           | Session Shard 创建的 Agentic 样本和 `rl.agent.run`，以及独立调用 shard 时创建的批次。                                                              |
| `agent`                           | 示例 agent 进程创建 `agent.invoke`、`llm.call` 和实际工具调用的 `tool.call`。自定义 agent 在自己的 SDK 中填写。                                    |
| `sglang`                          | SGLang 创建的请求和内部推理 span，包括健康检查。Relax 的 SGLang 镜像在 SDK 写入；旧镜像使用仓库附带的 OTLP 接收器时按 `sglang server` scope 补充。 |
| `actor_service`、`critic_service` | 发起分布式训练请求的服务，记录 `train_step`。                                                                                                      |
| `<role>_rank`                     | 训练 worker 注册时的实际角色，如 `actor_rank`、`critic_rank`、`actor_fwd_rank`。Actor rank 运行 reference forward 也保留 `actor_rank`。            |
| `advantages_service`              | 独立 Advantages 服务执行的优势计算。                                                                                                               |
| `driver` / `<role>_service`       | Driver 观测角色部署和重启；具体角色的 Serve 副本观测本进程的构造。                                                                                 |
| `framework`                       | 调用通用 Stage API 且未注册角色的框架进程。                                                                                                        |

例如 rank 0 观察到 rollout 权重加载确认时，`rl.rollout.load_weights` 的 `rl.producer.role=actor_rank`，`rl.backend.name=sglang` 描述加载目标。Traceparent 只传播调用上下文。

### 服务启动和重启（0.21）

按 `rl.run.id` 查询同一运行的服务生命周期。生命周期 trace 独立于 rollout、训练和 iteration。Driver 使用 `rl.producer.role=driver`；副本使用 `<role>_service`，Resource 记录实际主机和 PID。`rl.service.role` 表示被操作的逻辑角色，根重启 span 上的角色表示引发恢复的服务。

| Span                                                 | 记录边界与上下文                                                                                                                                                                                                                                                      |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `start services`                                     | Driver 进入和退出角色注册。首次启动是独立根；全局重启期间挂在本次 `rl.service.restart` 下。`rl.service.start.reason` 为 `initial` 或 `global_restart`；`rl.service.restart.global_count` 为 driver 的全局恢复计数。                                                   |
| `rl.service.prepare`                                 | 分阶段启动（S3 模型预取）时单个角色的资源准备任务。                                                                                                                                                                                                                   |
| `start <role>`                                       | 一个角色的实际部署调用。在常规启动中包含该任务内的资源准备；分阶段启动单独计部署。局部重启时挂在重启根下。记录角色、可用的配置 GPU 数 `rl.service.gpus`。到达绑定时设置 `rl.service.deployment.id`，值为本次 start 的 span ID。异常、任务返回失败或启动被中止均可见。 |
| `init <role>_service`、`enter <role>_service`        | 每次内置角色的 Serve 副本构造都有独立 init 根，以 Span Link 指向绑定它的 start。enter 子 span 在构造入口立即落盘，init 在构造返回或抛错时落盘。二者保留同一部署 ID。在进程退出或尚未完成时可能只有 enter。                                                            |
| `rl.service.restart`、`rl.service.restart.requested` | 每次 Controller 处理重启有独立根；决策后立即写 requested 子 span。记录触发角色、模式（`local`／`global`）、该角色的重启计数和决策原因。局部恢复记录能确认的 step 恢复和任务提交结果；全局恢复到 Controller 重新初始化完成。后续训练进度仍由训练 spans 表示。          |
| `rl.service.stop`、`rl.service.restore_step`         | stop 计一次角色部署删除调用，并在已知时记录旧部署 ID。restore_step 计局部恢复的 set_step 调用，记录目标 `rl.service.restore.step`；调用失败时本 span 和重启根都标为 ERROR，即使恢复逻辑继续提交任务。                                                                 |

### 问答：怎样区分首次启动、框架重启和 Serve 换副本？

展开 `start services` 可以看到 driver 部署每个角色的耗时。例如 `start actor` 对应 Actor 的部署调用，副本进程的 `init actor_service` 通过 Span Link 指回这次部署。找到 `rl.service.restart` 可以看到哪个角色触发恢复，哪些角色实际被删除和重新部署。全局重建时每个重新绑定的角色有新的 deployment ID。在同一 run、角色和 deployment ID 下，多个 `enter <role>_service` 表示同一绑定发生了多次构造；根据各自的 Resource 和 init 状态区分进程、重试和已完成的构造。单凭多次入口无法确定底层故障原因。采集覆盖 Relax 注册的角色服务；角色内部的 Megatron rank、SGLang 引擎及独立基础设施各有自己的生命周期。

### 问答：Megatron 原生 span 如何接入训练链路？

训练 RPC 执行期间，rank 将自己的 span 设为当前 OTel 上下文。进入已埋点的前向或训练阶段时，当前上下文切换到该阶段，离开时恢复。因此重新计算 logprob 时导出的 `megatron.microbatch.forward` 属于正在执行的 `rl.actor.forward` 或 `rl.reference.forward`；直接由 RPC 在阶段之外生成的 span 继承该 rank。启动进程等发生在训练请求之前的操作有自己的生命周期。

同时启用框架 RL tracing 和 Megatron Lens 时，实际导出的 `relax.train.step` 继承本 rank 的 `rl.actor.train` 或 `rl.critic.train` 的 trace ID，并以该操作作为直接父节点。该 step 对应一次优化器更新，内部的前反向和优化器 span 由 Lens 的 span group 与 rank 采样策略控制。Lens Resource 中的 `rl.run.id` 与 `rl.producer.role` 标记框架运行及产出进程。Lens 的运行标识和导出目标仍由实验配置决定。按 trace ID 和 parent span ID 可跨两个导出端还原调用树。

### 问答：一个训练请求如何对应多个 Megatron 进程？

展开 `rl.actor.train_step` 或 `rl.critic.train_step`，每个实际收到请求的 rank 都有一个同 trace 的直接子 span。按 `rl.rank.global` 选 rank，按 OTel Resource 的 `host.name`、`process.pid` 找到实际 worker。没有执行计算的 rank 仍有 RPC span，只展开它实际执行的阶段。训练组返回原有 Ray ObjectRef；框架不会收集各 rank 的阶段报告来重建时间线。异常在发生的层级记录；worker 进程硬退出前尚未结束的 span 可能缺失。

同一训练请求的各 rank 共享 trace ID，每个 rank 和阶段都有独立的 span ID。框架从操作系统随机源分配新 ID，即使多个 rank 使用同一训练种子也能按父子 ID 还原，埋点不会消耗训练用的 Python 随机序列。

训练 RPC 中实际发起权重同步时，各参与 rank 用 `rl.weight.sync` 记录自己的同步调用。Hybrid 模式的 `update_weights` 委托 DCS 执行，同一 rank 的这次调用只产生一个 sync。独立于训练请求的同步由发起侧 rank 0 记录一个 sync 根。

在使用 Tensor 或 Distributed 权重更新器的路径中，发送端 rank 0 的 `rl.actor.publish_weights` 和 `rl.rollout.load_weights` 直接挂在本次 sync 下，并保留交付 Link。DCS 路径当前记录同步方法的耗时。查询每个接收端的参数版本和激活耗时需要该路径提供相应的明细埋点。FSDP 和独立提交的前向服务沿用各自入口。

## 3 核心阶段 span

### Span: `rl.rollout.generate`

**Status:** Development

该 span 对应框架发起的一次 rollout 调用，记录这一调用接纳的所有样本。每个样本使用 `rl.rollout.sample`，通过 `rl.group.id` 标出所属 prompt group。这次调用发起的评分批次、动态过滤、队列写入、Agentic 自定义优势计算和 OPD teacher／student prefill 直接挂在它下面；各操作以实际发生次数计。普通和 Agentic rollout 使用相同的调用归属规则。一个 step 可以发起多次 rollout 调用。

从框架调用 rollout 函数到该函数返回、失败或取消，包含函数内实际执行或等待的生成、评分、组后处理以及队列交付。并发工作按实际墙钟时间记录。后续训练和调用方的过滤不计入。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.rollout.generate`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                              | Value Type | Description                                                        | Example Values     |
| -------------------- | ----------- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------ | ------------------ |
| `rl.run.id`          | Development | Required                                       | string     | 本次实际运行的唯一身份。                                           | `run-a7`           |
| `rl.purpose`         | Development | Required                                       | string     | 操作用途：training 或 evaluation。                                 | `training`         |
| `rl.batch.id`        | Development | Required                                       | string     | 本窗口接纳的生成任务集合身份；不沿用为后续训练批次身份。           | `rollout-batch-17` |
| `rl.batch.size`      | Development | Required：输入集合确定时                       | int        | 实际接纳的输入任务数，排除 padding，不按 rank 重复计数。           | `128`              |
| `rl.rollout.mode`    | Development | Required                                       | string     | 生成方式：single_turn、multi_turn、agentic。                       | `agentic`          |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                         | `learner:80`       |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                     | `80`               |
| `rl.model.name`      | Development | Conditionally Required：本窗口使用唯一模型     | string     | 当前操作使用的模型名称；多模型窗口在各请求数据中分别记录。         | `policy-7b`        |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                         | `sglang`           |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照   | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。 | `snapshot-48`      |

普通部分 rollout 窗口可以在样本暂停时结束，完整 sample 保持未结束；窗口 Status 按交付承诺判断。Agentic 按一次框架调用聚合多个 Group。每个 Session 的 `rl.agent.run` 以该 Session 的 `rl.rollout.sample` 为 parent。

### Span: `rl.rollout.sample`

**Status:** Development

该 span 表示每条题目的一次完整 rollout。普通模式包含该输入的全部生成调用；Agentic 模式对应一个 Session，包含所有模型／工具交互以及输出构建。一个 Sample 可以产出多个 Export，不按 Export 数重复创建该 span。

从首次接纳此输入开始，到生成结果完成或不可恢复的失败／取消。可恢复中断和暂停不结束。已有完成结果被再次读取时不新建 span。parent 固定为首次接纳的 `rl.rollout.generate`，恢复窗口按第 2.1 节关联。

**Span kind:** `INTERNAL`。**Span name:** 固定为 `rl.rollout.sample`。

**Span status:** 按第 2 章；已开始但失败且没有 Export 时也必须结束并记录错误。

**Requirement level:** `Required`；实际生成一个输入时采集。

除第 2.1 节公共属性，还记录：

| Key                                                    | Stability   | Requirement Level                         | Value Type | Description                                                                            |
| ------------------------------------------------------ | ----------- | ----------------------------------------- | ---------- | -------------------------------------------------------------------------------------- |
| `rl.run.id`                                            | Development | Required                                  | string     | 本次运行身份                                                                           |
| `rl.purpose`                                           | Development | Required                                  | string     | training 或 evaluation                                                                 |
| `rl.rollout.mode`                                      | Development | Required                                  | string     | single_turn、multi_turn 或 agentic                                                     |
| `rl.group.id`                                          | Development | Conditionally Required：prompt group 已知 | string     | 同一 prompt 本次重复采样的分组身份；每个样本仍拥有独立的 sample ID。                   |
| `rl.session.id`                                        | Development | Conditionally Required：Agentic           | string     | 对应完整 Session 身份                                                                  |
| `rl.sample.input`                                      | Development | Required：输入可取得时                    | string     | 完整原始题目／messages 的 JSON；保持文本或消息结构，不填格式化后的猜测                 |
| `rl.sample.response`                                   | Development | Conditionally Required：已有实际输出      | string     | JSON 数组，每项保存 export_id、可取得的 name 及 text 或 messages；多 Export 不任选一条 |
| `rl.model.name`、`rl.backend.name`、`rl.model.version` | Development | Conditionally Required：唯一且已知        | string     | 本次真实执行的模型、后端和参数快照                                                     |

普通和 Agentic 的样本 span 在来源已知时记录 `rl.prompt.id`。一次采样的所有 Export 保留同一题目身份。普通 rollout 的 response 保存实际生成文本和框架状态 status；取消／失败时保留已有部分文本，`rl.sample.cancelled=true` 表示已确认丢弃，不把部分文本冒充完整答案。Agentic response 由各 Export 的状态树分支取得最后的 assistant 消息，不把混合工具观察的训练 continuation 冒充最终回答。完整轨迹继续记录在 `rl.agent.run`。失败／无输出时省略 response，不用空字符串表示未知。

### Span: `rl.reward.batch`

**Status:** Development

该 span 是一批数据评分的总览，包含实际逐条 `rl.reward.score` 子 span。输入流式到达时，批次按已接纳的 rollout 窗口／明确的评分调用集合界定，使用新的评分 batch.id；batch.size 只计实际启动评分的输入，读取已有 reward 不计入。

一次 rollout 可以包含零个、一个或多个评分批次。每个批次创建时使用发起它的 `rl.rollout.generate` 作为直接 parent，共享 traceId；独立调用评分器且无 rollout 上下文时使用根 span。Agentic 跨轮恢复时，在新轮次开始的评分批次归新轮次，已经开始的批次保留原 parent。输入来自哪个样本单独通过 Link 表达。

从首个评分开始，到输入窗口封口且所有已启动评分结束。末尾无评分的生成／后处理不计入原生结束时间；一个子评分提前失败不能关闭仍运行的兄弟评分。真正的批量 scorer 只生成一次批次 span，成员结果直接记录在 rl.reward.results，不生成 score 子 span。

**Span kind:** `INTERNAL`。**Span name:** 固定为 `rl.reward.batch`。

**Span status:** 未处理的评分失败为 ERROR，正常零分／负分保持 UNSET。

**Requirement level:** `Required`；有实际评分时采集。

属性为第 2.1 节批次公共属性和 `rl.reward.score` 中的 run、purpose、step、评分方案、模型属性。 批次结果保存最终返回值；若批量方案在独立子调用后再聚合或变换分数，子 span 保留该次调用的原始返回值，变换后的最终值和下游来源属于批次，不能用中间分数覆盖最终结果。混合评分路由使用完整方案名；批次 `rl.reward.results` 保存完整逐成员结果，子 span 保存自己对应的结果，不以平均值替代明细。

### Span: `rl.reward.score`

**Status:** Development

该 span 表示对一条已有轨迹的一次独立评分调用，包括规则、验证器、奖励模型或模型裁判。仅针对真实独立评分调用创建，parent 为实际 `rl.reward.batch`，使用 operation 时间；批量 scorer 返回的逐成员结果保存在批次属性中。

从评价逻辑开始，到分数及有效性结果可供下游使用。读取已有 reward 不代表再次评分，不生成虚假的评分耗时。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.reward.score`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                                  | Value Type | Description                                                        | Example Values                                                                                                        |
| -------------------- | ----------- | -------------------------------------------------- | ---------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `rl.run.id`          | Development | Required                                           | string     | 本次实际运行的唯一身份。                                           | `run-a7`                                                                                                              |
| `rl.purpose`         | Development | Required                                           | string     | 操作用途：training 或 evaluation。                                 | `training`                                                                                                            |
| `rl.reward.name`     | Development | Required                                           | string     | 本次独立评分调用的方案／评分器名称。                               | `math_verifier`                                                                                                       |
| `rl.reward.type`     | Development | Required                                           | string     | rule、verifier、model、judge、environment 或 hybrid。              | `verifier`                                                                                                            |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step          | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                         | `learner:80`                                                                                                          |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次           | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                     | `80`                                                                                                                  |
| `rl.model.name`      | Development | Conditionally Required：采用模型评分且使用单个模型 | string     | 当前操作使用的模型名称，不用算法名称代替。                         | `reward-model`                                                                                                        |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一     | string     | 实际模型执行后端；Ray 不属于模型执行后端。                         | `megatron`                                                                                                            |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照       | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。 | `snapshot-48`                                                                                                         |
| `rl.reward.results`  | Development | Required：对应结果可取得时                         | string     | 逐 Export 的评分结果。\[1\] 值为完整 JSON 字符串。                 | `[{"prompt_id":"prompt_…","sample_id":"s1","export_id":"export-1","source":"verifier","status":"valid","value":2.0}]` |

**\[1\] `rl.reward.results`:** 来源已知时，逐条评分 span 记录 `rl.prompt.id`，评分明细（含批量、失败及无效结果）使用 `prompt_id`。同一评分批次可以有多个题目 ID，各结果行分别记录。每项包含 export_id、source、status（valid／invalid／error）、标量或复合 value；复合值明确标量选择口径，失败／无效时记录 reason。不能只保存批次平均值。零分、负分或答案错误可以是成功评分；多个独立评分调用组成方案时，`rl.reward.batch` 表示完整批次，`rl.reward.score` 只表示本次实际调用；聚合或变换后的最终值放在 batch 上。

### Span: `rl.reference.forward`

**Status:** Development

该 span 表示参考策略对已有 token 执行无梯度前向，提供 reference logprob，用于 KL 等约束。

从开始独立参考前向，到参考概率输出可用。临时关闭 adapter 或切换参考权重不改变此逻辑职责。

按实际批次／DP 分区调用记录一个 span，成员结果保存在 `rl.forward.results`，不生成 `rl.reference.forward.sample`。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.reference.forward`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                              | Value Type | Description                                                                    | Example Values                             |
| -------------------- | ----------- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------------------ | ------------------------------------------ |
| `rl.run.id`          | Development | Required                                       | string     | 本次实际运行的唯一身份。                                                       | `run-a7`                                   |
| `rl.model.name`      | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                                     | `reference-7b`                             |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                                     | `learner:80`                               |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                                 | `80`                                       |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                                     | `megatron`                                 |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照   | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。             | `snapshot-48`                              |
| `rl.forward.results` | Development | Required：对应结果可取得时                     | string     | 完整 JSON 数组；保存真实 export_id 或 sample_index 及结果状态，详见第 2.1 节。 | `[{"sample_index":"17","status":"valid"}]` |

### Span: `rl.teacher.forward`

**Status:** Development

该 span 表示教师对已有学生轨迹执行评分前向，提供在线蒸馏的概率目标。

从开始教师评分前向，到所需概率目标可用。教师另行生成答案时属于生成调用，不能记成固定轨迹 forward。

按实际批次／DP 分区调用记录一个 span，成员结果保存在 `rl.forward.results`，不生成 `rl.teacher.forward.sample`。 OPD 单样本前向同样只记录一个真实调用 span，`rl.batch.size=1`。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.teacher.forward`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                              | Value Type | Description                                                                    | Example Values                             |
| -------------------- | ----------- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------------------ | ------------------------------------------ |
| `rl.run.id`          | Development | Required                                       | string     | 本次实际运行的唯一身份。                                                       | `run-a7`                                   |
| `rl.model.name`      | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                                     | `teacher-32b`                              |
| `rl.forward.output`  | Development | Required                                       | string     | 教师目标类型：sampled_token_logprob、topk_distribution、full_distribution。    | `topk_distribution`                        |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                                     | `learner:80`                               |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                                 | `80`                                       |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                                     | `megatron`                                 |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照   | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。             | `snapshot-48`                              |
| `rl.forward.results` | Development | Required：对应结果可取得时                     | string     | 完整 JSON 数组；保存真实 export_id 或 sample_index 及结果状态，详见第 2.1 节。 | `[{"sample_index":"17","status":"valid"}]` |

教师概率目标与 reference 约束、原始 reward 是不同职责，不能因接口复用而混同。

### Span: `rl.critic.forward`

**Status:** Development

该 span 表示更新前的价值模型对已有轨迹估值，为优势和 returns 构造提供旧 values。

从开始独立价值预测，到 values 可用。价值模型训练内部的带梯度前向属于 critic.train。

按实际批次／DP 分区调用记录一个 span，成员结果保存在 `rl.forward.results`，不生成 `rl.critic.forward.sample`。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.critic.forward`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                              | Value Type | Description                                                                    | Example Values                             |
| -------------------- | ----------- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------------------ | ------------------------------------------ |
| `rl.run.id`          | Development | Required                                       | string     | 本次实际运行的唯一身份。                                                       | `run-a7`                                   |
| `rl.model.name`      | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                                     | `critic-7b`                                |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                                     | `learner:80`                               |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                                 | `80`                                       |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                                     | `megatron`                                 |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照   | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。             | `snapshot-48`                              |
| `rl.forward.results` | Development | Required：对应结果可取得时                     | string     | 完整 JSON 数组；保存真实 export_id 或 sample_index 及结果状态，详见第 2.1 节。 | `[{"sample_index":"17","status":"valid"}]` |

算法未使用 Critic 时不生成此 span。

### Span: `rl.advantage.compute`

**Status:** Development

该 span 表示根据奖励、概率和 values 等构造优势及按需产生的 returns，包含归属该过程的整形与归一化。

从依赖输入可用并开始目标构造，到优势／returns 可被训练消费。融合在训练中的计算仍单独记录阶段 span，按第 2 章选择复用或独立 trace，与训练 span 不建立父子关系；由消费其结果的训练 span 通过 Span Link 关联，时间边界保留真实重叠，不重复累计耗时。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.advantage.compute`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                      | Stability   | Requirement Level                         | Value Type | Description                                        | Example Values                                            |
| ------------------------ | ----------- | ----------------------------------------- | ---------- | -------------------------------------------------- | --------------------------------------------------------- |
| `rl.run.id`              | Development | Required                                  | string     | 本次实际运行的唯一身份。                           | `run-a7`                                                  |
| `rl.advantage.estimator` | Development | Required                                  | string     | 实际估计方法；开放枚举，如 grpo、gae、rloo、opd。  | `grpo`                                                    |
| `rl.step.id`             | Development | Conditionally Required：明确唯一所属 Step | string     | Run 内的 Step 身份；跨 Step 时不任选一个。         | `learner:80`                                              |
| `rl.iteration.index`     | Development | Conditionally Required：明确唯一外层轮次  | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。     | `80`                                                      |
| `rl.advantage.results`   | Development | Required：对应结果可取得时                | string     | 逐 Export 的优势摘要。\[1\] 值为完整 JSON 字符串。 | `[{"export_id":"export-1","mean":0.0,"token_count":158}]` |

**\[1\] `rl.advantage.results`:** 每项包含真实 export_id 或 sample_index，以及可取得的 mean、token_count。均值按该阶段有效位置计算，无有效位置时省略 mean；需要展示原始分项时保留 values 及其粒度。实际训练 mask 发生变化时，训练结果使用最终 mask 对应的摘要。这些统计是批次结果记录，不生成 `rl.advantage.compute.sample`。

### Span: `rl.actor.forward`

**Status:** Development

该 span 表示策略对已有轨迹独立执行无梯度前向，准备固定概率基准或学生蒸馏概率。

从开始独立策略评分，到所需概率输出可用。训练内部带梯度前向不属于此操作；直接复用 rollout logprob 时不创建此 span。

按实际批次／DP 分区调用记录一个 span，成员结果保存在 `rl.forward.results`，不生成 `rl.actor.forward.sample`。 OPD 单样本前向同样只记录一个真实调用 span，`rl.batch.size=1`。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.actor.forward`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                              | Value Type | Description                                                                    | Example Values                             |
| -------------------- | ----------- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------------------ | ------------------------------------------ |
| `rl.run.id`          | Development | Required                                       | string     | 本次实际运行的唯一身份。                                                       | `run-a7`                                   |
| `rl.model.name`      | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                                     | `policy-7b`                                |
| `rl.forward.purpose` | Development | Required                                       | string     | 独立策略评分的用途：old_policy 或 distillation。                               | `old_policy`                               |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                                     | `learner:80`                               |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                                 | `80`                                       |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                                     | `megatron`                                 |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照   | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。             | `snapshot-48`                              |
| `rl.forward.results` | Development | Required：对应结果可取得时                     | string     | 完整 JSON 数组；保存真实 export_id 或 sample_index 及结果状态，详见第 2.1 节。 | `[{"sample_index":"17","status":"valid"}]` |

### Span: `rl.actor.train`

**Status:** Development

该 span 表示对有界批次或流式输入窗口执行策略训练，包含带梯度前向、loss、反向、梯度累积和优化器更新决策。

从接纳本次训练输入／窗口并开始训练，到该窗口更新流程完成，或失败／取消处理结束。Relax 的每个参与 rank 从本地实际 train 调用入口开始，调用内流式取数等待计入 host 时间；RPC 入口、调用前准备／预取（包括 dynamic+VPP 的整桶预取）、权重发布、checkpoint 和训练后 offload 不在此边界内。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.actor.train`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                       | Stability   | Requirement Level                              | Value Type | Description                                                                | Example Values                                                                                                                                            |
| ------------------------- | ----------- | ---------------------------------------------- | ---------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rl.run.id`               | Development | Required                                       | string     | 本次实际运行的唯一身份。                                                   | `run-a7`                                                                                                                                                  |
| `rl.batch.id`             | Development | Required                                       | string     | 本次策略训练实际接纳的样本集合身份，与生成任务批次分开标识。               | `train-batch-80`                                                                                                                                          |
| `rl.batch.size`           | Development | Required：输入集合确定时                       | int        | 本次策略训练接纳的训练样本数；排除 padding，不按 epoch 或 rank 重复计数。  | `512`                                                                                                                                                     |
| `rl.model.name`           | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                                 | `policy-7b`                                                                                                                                               |
| `rl.algorithm.name`       | Development | Required                                       | string     | 当前执行的训练算法；开放枚举。                                             | `grpo`                                                                                                                                                    |
| `rl.step.id`              | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                                 | `learner:80`                                                                                                                                              |
| `rl.iteration.index`      | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                             | `80`                                                                                                                                                      |
| `rl.backend.name`         | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                                 | `megatron`                                                                                                                                                |
| `rl.model.version.before` | Development | Conditionally Required：开始版本已知           | string     | 开始训练时实际使用的参数快照。                                             | `snapshot-48`                                                                                                                                             |
| `rl.model.version.after`  | Development | Conditionally Required：结束时一致版本可确认   | string     | 实际结束版本；全部更新跳过时可与 before 相同，失败后不确定则省略。         | `snapshot-49`                                                                                                                                             |
| `rl.train.results`        | Development | Required：对应结果可取得时                     | string     | 逐 Update 结果及逐 Export 的实际训练消费关系。\[1\] 值为完整 JSON 字符串。 | `{"updates":[{"update_id":"update-1","status":"committed"}],"uses":[{"export_id":"export-1","update_id":"update-1","used":true,"effective_tokens":158}]}` |

**\[1\] `rl.train.results`:** 可取得时包含 updates 和 uses 两个列表；rank 只写自身可观测的结果。updates 保存 update_id、已确认的 status（committed／skipped／failed）、reason，以及可获得的后端 span_context、source_name、start_time、end_time。uses 保存每个成员与 update_id 的实际消费关系，成员使用真实 export_id 或 sample_index 标识，并记录 used、最终 effective_tokens 和可取得的 advantage_mean，保留多 epoch 的多次消费关系。uses 和 updates 都是 JSON 结果记录，不生成逐样本 train span；共享更新结果不代表每个样本有独立的训练耗时。

更新是否提交取自真实决策，不能由训练函数正常返回推断。used=true、有效 token 大于 0、参数更新已提交是不同事实。时间只引用已有后端记录，不新增 optimizer span，也不能把 train_batch 整段冒充纯 optimizer 耗时。

### Span: `rl.critic.train`

**Status:** Development

该 span 表示使用 returns 等目标训练价值模型，包含价值模型的带梯度前向、loss、反向和更新决策。

从接纳价值训练输入／窗口并开始训练，到本窗口价值更新流程完成，或失败／取消处理结束。独立旧 values 预测属于 critic.forward；Relax 从实际 train 调用入口开始，调用内流式等待计入 host 时间，调用前数据准备与调用后 checkpoint／offload 不计入。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.critic.train`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                       | Stability   | Requirement Level                              | Value Type | Description                                                                | Example Values                                                                                                                                            |
| ------------------------- | ----------- | ---------------------------------------------- | ---------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rl.run.id`               | Development | Required                                       | string     | 本次实际运行的唯一身份。                                                   | `run-a7`                                                                                                                                                  |
| `rl.batch.id`             | Development | Required                                       | string     | 本次价值训练实际接纳的样本集合身份，与生成任务批次分开标识。               | `train-batch-80`                                                                                                                                          |
| `rl.batch.size`           | Development | Required：输入集合确定时                       | int        | 本次价值训练接纳的训练样本数；排除 padding，不按 epoch 或 rank 重复计数。  | `512`                                                                                                                                                     |
| `rl.model.name`           | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                                 | `critic-7b`                                                                                                                                               |
| `rl.step.id`              | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                                 | `learner:80`                                                                                                                                              |
| `rl.iteration.index`      | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                             | `80`                                                                                                                                                      |
| `rl.backend.name`         | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                                 | `megatron`                                                                                                                                                |
| `rl.model.version.before` | Development | Conditionally Required：开始版本已知           | string     | 开始训练时实际使用的参数快照。                                             | `snapshot-48`                                                                                                                                             |
| `rl.model.version.after`  | Development | Conditionally Required：结束时一致版本可确认   | string     | 实际结束版本；全部更新跳过时可与 before 相同，失败后不确定则省略。         | `snapshot-49`                                                                                                                                             |
| `rl.train.results`        | Development | Required：对应结果可取得时                     | string     | 逐 Update 结果及逐 Export 的实际训练消费关系。\[1\] 值为完整 JSON 字符串。 | `{"updates":[{"update_id":"update-1","status":"committed"}],"uses":[{"export_id":"export-1","update_id":"update-1","used":true,"effective_tokens":158}]}` |

**\[1\] `rl.train.results`:** 可取得时包含 updates 和 uses 两个列表；rank 只写自身可观测的结果。updates 保存 update_id、已确认的 status（committed／skipped／failed）、reason，以及可获得的后端 span_context、source_name、start_time、end_time。uses 保存每个成员与 update_id 的实际消费关系，成员使用真实 export_id 或 sample_index 标识，并记录 used、最终 effective_tokens 和可取得的 advantage_mean，保留多 epoch 的多次消费关系。uses 和 updates 都是 JSON 结果记录，不生成逐样本 train span；共享更新结果不代表每个样本有独立的训练耗时。

更新是否提交取自真实决策，不能由训练函数正常返回推断。used=true、有效 token 大于 0、参数更新已提交是不同事实。时间只引用已有后端记录，不新增 optimizer span，也不能把 train_batch 整段冒充纯 optimizer 耗时。

### Span: `rl.actor.publish_weights`

**Status:** Development

该 span 表示将一份一致的策略快照交付给约定接收端，使其可以获取该版本。

从开始序列化、发布或传输，到源端交付完成：拉取式为快照已可读取，推送式为约定目标已确认接收。此边界不包含目标激活。若适配器只能提供源端同步调用边界，必须标明 `rl.timing.scope=source_sync`，遵循第 2 章的边界说明，不能标为纯交付耗时。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.actor.publish_weights`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                     | Stability   | Requirement Level                              | Value Type | Description                                                        | Example Values |
| ----------------------- | ----------- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------ | -------------- |
| `rl.run.id`             | Development | Required                                       | string     | 本次实际运行的唯一身份。                                           | `run-a7`       |
| `rl.model.name`         | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                         | `policy-7b`    |
| `rl.model.version`      | Development | Required                                       | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。 | `snapshot-48`  |
| `rl.weight.transfer.id` | Development | Required                                       | string     | 一次快照交付的身份，与对应 load_weights 使用相同值。               | `transfer-49`  |
| `rl.step.id`            | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                         | `learner:80`   |
| `rl.iteration.index`    | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                     | `80`           |
| `rl.backend.name`       | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                         | `megatron`     |

仅提交异步发送请求不表示发布完成。发布成功不表示所有生成副本已经激活。

### Span: `rl.rollout.load_weights`

**Status:** Development

该 span 表示指定生成目标获取、安装并激活已发布的策略快照。

从目标开始获取或安装参数，到本次声明的目标范围确认生效。单个 rank 安装完成不能代替整个目标可用的确认。Relax 当前仅有源端同步边界，使用 `rl.timing.scope=source_sync`；起点包含准备过程，不声称观测到目标实际安装起点。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.rollout.load_weights`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                     | Stability   | Requirement Level                              | Value Type | Description                                                        | Example Values      |
| ----------------------- | ----------- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------ | ------------------- |
| `rl.run.id`             | Development | Required                                       | string     | 本次实际运行的唯一身份。                                           | `run-a7`            |
| `rl.model.name`         | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                         | `policy-7b`         |
| `rl.model.version`      | Development | Required                                       | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。 | `snapshot-48`       |
| `rl.weight.transfer.id` | Development | Required                                       | string     | 与对应发布操作相同的快照交付身份。                                 | `transfer-49`       |
| `rl.weight.target.id`   | Development | Required                                       | string     | 本次承诺激活的副本或目标集合身份；集合成员必须明确。               | `rollout-replica-2` |
| `rl.step.id`            | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                         | `learner:80`        |
| `rl.iteration.index`    | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                     | `80`                |
| `rl.backend.name`       | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                         | `sglang`            |

CPU/GPU 驻留恢复但参数版本未变化，不属于新版本加载。

### Span: `rl.data.filter`

**Status:** Development

该 span 表示对一批轨迹执行过滤，决定是否进入后续训练流程。

从开始过滤判断，到本次 keep/drop 决策完成。正常 drop 是判定成功，不自动设置 ERROR。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.data.filter`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                         | Value Type | Description                                              | Example Values                                 |
| -------------------- | ----------- | ----------------------------------------- | ---------- | -------------------------------------------------------- | ---------------------------------------------- |
| `rl.run.id`          | Development | Required                                  | string     | 本次实际运行的唯一身份。                                 | `run-a7`                                       |
| `rl.filter.name`     | Development | Required                                  | string     | 实际执行的过滤方案名称。                                 | `reward_validity`                              |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step | string     | Run 内的 Step 身份；跨 Step 时不任选一个。               | `learner:80`                                   |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次  | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。           | `80`                                           |
| `rl.group.id`        | Development | Conditionally Required：对单个 Group 判定 | string     | 本次被过滤的 Group 身份。                                | `group-159`                                    |
| `rl.filter.results`  | Development | Required：对应结果可取得时                | string     | 每个输入 Export 的过滤结果。\[1\] 值为完整 JSON 字符串。 | `[{"export_id":"export-1","decision":"keep"}]` |

**\[1\] `rl.filter.results`:** 每项包含真实 export_id 或 sample_index、decision（keep／drop）；drop 时包含 reason。整组过滤也保留每个成员身份，不能只记录组级数量。这些决定保存在批次属性中，不生成 `rl.data.filter.sample`。

### Span: `rl.data.transfer`

**Status:** Development

该 span 表示将一批轨迹交付训练数据系统，提供传输耗时、成员和完成状态。

从本批次开始转换／发送，到实际接收确认，或失败／取消处理结束。开始发送、调用结束与接收成功是不同事实。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.data.transfer`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                          | Stability   | Requirement Level                         | Value Type | Description                                          | Example Values                |
| ---------------------------- | ----------- | ----------------------------------------- | ---------- | ---------------------------------------------------- | ----------------------------- |
| `rl.run.id`                  | Development | Required                                  | string     | 本次实际运行的唯一身份。                             | `run-a7`                      |
| `rl.transfer.target`         | Development | Required                                  | string     | 实际目标 partition／队列。                           | `train_80`                    |
| `rl.step.id`                 | Development | Conditionally Required：明确唯一所属 Step | string     | Run 内的 Step 身份；跨 Step 时不任选一个。           | `learner:80`                  |
| `rl.iteration.index`         | Development | Conditionally Required：明确唯一外层轮次  | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。       | `80`                          |
| `rl.transfer.target.step_id` | Development | Conditionally Required：目标 Step 已知    | string     | 实际交付目标 Step，可以与轨迹生成 Step 不同。        | `learner:80`                  |
| `rl.transfer.data`           | Development | Required：对应结果可取得时                | string     | 批次成员及已知传输边界。\[1\] 值为完整 JSON 字符串。 | `{"export_ids":["export-1"]}` |

**\[1\] `rl.transfer.data`:** 保存 export_ids，以及已知的入 buffer、入队、发送开始和接收确认时间；members 可保存各成员的真实 export_id、sample_index。共享发送／ACK 边界只记录在批次上，不为每个成员复制时间或生成 `rl.data.transfer.sample`。如已有真实成员边界，以成员身份标明其归属。结束边界缺失时保持缺失，不能根据后续训练成功反推传输完成。

## 4 Agent 调用 span

### Span: `rl.agent.run`

**Status:** Development

该 span 表示一次 Agent 调用及其产出的轨迹。一个 Session 对应一个 Agent span，包含多轮模型调用、工具交互、状态树及多个 Export。失败且没有 Export 的调用也必须保留。

从 Agent 启动，到调用终态、Export 构建和结果交付结束。轨迹中的 execution 单独记录 Agent 实际执行起止，区分执行时间与导出／清理时间。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.agent.run`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Parent:** 对应 Session 的 `rl.rollout.sample`，与发起本次 rollout 的 `rl.rollout.generate` 共享 traceId。暂停恢复保留同一个 Agent span 和 traceId，跨轮保留实际 Step 归属。预取阶段提前终止且从未分配的调用仍有 sample 和批次，必须标记 unassigned。

**Attributes:**

| Key                          | Stability   | Requirement Level                               | Value Type | Description                                                     | Example Values |
| ---------------------------- | ----------- | ----------------------------------------------- | ---------- | --------------------------------------------------------------- | -------------- |
| `rl.run.id`                  | Development | Required                                        | string     | 本次实际运行的唯一身份。                                        | `run-a7`       |
| `rl.purpose`                 | Development | Required                                        | string     | 操作用途：training 或 evaluation。                              | `training`     |
| `rl.session.id`              | Development | Required                                        | string     | 一次 Agent 调用的稳定身份，暂停恢复保持不变。                   | `session-1`    |
| `rl.agent.name`              | Development | Required                                        | string     | Agent／runner 的稳定名称。                                      | `visual-agent` |
| `rl.rollout.assignment`      | Development | Required：结束时                                | string     | assigned 或 unassigned；未被分配即终止的预取调用为 unassigned。 | `assigned`     |
| `rl.session.turn_count`      | Development | Required：结束时                                | int        | 已提交模型响应的轮次数，同一逻辑请求重试不新增一轮。            | `2`            |
| `rl.session.export_count`    | Development | Required：结束时                                | int        | 成功生成的不同 Export 数，无导出时为 0。                        | `1`            |
| `rl.step.id`                 | Development | Conditionally Required：结束时确认唯一所属 Step | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                      | `learner:80`   |
| `rl.iteration.index`         | Development | Conditionally Required：结束时确认唯一外层轮次  | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                  | `80`           |
| `rl.group.id`                | Development | Conditionally Required：属于成组采样            | string     | 本次成组采样身份；相同题目再次采样使用新 Group。                | `group-159`    |
| `rl.group.expected_sessions` | Development | Conditionally Required：预期组大小已知          | int        | Group 的预期 Agent 调用数量。                                   | `4`            |
| `rl.session.output_tokens`   | Development | Conditionally Required：完整用量可确认          | int        | 逻辑模型请求最终输出的去重 token 总数，不含工具观察和重复前缀。 | `158`          |
| `rl.trajectory.data`         | Development | Required：对应结果可取得时                      | string     | 完整轨迹内容。\[1\] 值为完整 JSON 字符串。                      | 见字段说明     |

**\[1\] `rl.trajectory.data`:** 以下为 JSON 内容结构，不增加 span 类型。Request、Attempt、工具、状态节点和 Export 默认作为本次 Agent 调用的数据保存。

| 数据项                  | 必须保存的内容                                                                                                                                       |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `execution`             | Agent 实际执行的起止、结果、原因；结束缺失时保持缺失                                                                                                 |
| `input`                 | 原始题目／messages、label、Session 默认 sampling_config、tools 配置                                                                                  |
| `requests`              | 每个 Request 的 ID、接纳序号、模型、逻辑起止与结果、完整输入／输出消息、实际 sampling_config、token 用量、finish reason、parent／response state hash |
| `requests[].attempts`   | 独立 Attempt 序号、后端请求 ID、起止、结果／中断原因、实际模型版本、生成所属 Step、原始后端 metadata 及其单位；多次尝试不能只保留最后一次            |
| `requests[].milestones` | 真实发生的输入就绪、组内首请求条件满足、Runtime 接管、生成放行、本地派发、后端调用／返回、服务响应就绪等时间点及对应 Attempt；不适用或未知时省略     |
| `states`                | 所有已提交状态的 hash、parent hash、节点类型、messages_delta 和关联 Request；不仅保存最后导出的路径                                                  |
| `tools`                 | tool_call ID、名称、参数、返回消息及已有媒体摘要／引用；包括没有下一次模型请求的末尾工具返回；真实执行时间仅在可取得时提供                           |
| `exports`               | 每个 Export 的独立 ID、名称、构建起止与结果、leaf state hash、响应序列 token 数；Agent 已返回 reward 时同时保存 reward 及来源                        |
| `step_memberships`      | Session／Request／Attempt 的实际 Step 归属；区分计划归属与实际生成，不以最后一个 rollout ID 覆盖历史                                                 |
| `external_links`        | 已提供的 Session 外部详情链接，如 Langfuse；未提供时省略                                                                                             |

每个 Export 独立分配 ID，不能直接使用可能被多个 Export 共享的 sample.index。并发请求按稳定身份和状态 hash 连接，接纳序号仅用于排序。Agent 和 sample span 结束后不再追加后续评分、过滤、传输或训练结果；各阶段通过 Export ID 关联。

跨 Step 的 Agent span 省略单值 step.id／iteration.index，使用轨迹内归属关系和实际批次与样本关联。只观察到服务响应就绪时不能记为客户端已消费完整响应；工具真实执行时间或后端边界未知时不补算。
