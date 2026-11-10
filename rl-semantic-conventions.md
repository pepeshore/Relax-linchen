# 大模型在线 RL 语义规范草案

**Status:** Development

**Version:** 0.10.0

**Date:** 2026-11-10

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

阶段 span 由掌握完整操作生命周期的一方记录一次，耗时使用原生起止时间计算，包括该职责内部的通信、等待和重试。提交成功不代表操作完成；不同 rank 的记录不能累加成一次逻辑阶段的整体耗时。未执行的阶段不生成占位 span。

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

| 属性                             | 记录位置                                                 | 口径                                                                                   |
| -------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `rl.batch.id`、`rl.batch.size`   | 所有批次操作 span                                        | 各阶段记录自身实际输入集合身份与成员数；明细只记录所属 batch.id，不重复记录 batch.size |
| `rl.purpose`                     | `rl.rollout.generate`、`rl.reward.score`、`rl.agent.run` | 区分训练采样与评估；训练及训练准备阶段的用途已由职责确定，权重操作可能同时服务多个用途 |
| `rl.algorithm.name`              | `rl.actor.train`                                         | 记录本轮策略训练算法；优势阶段单独记录实际 estimator，Critic 不重复复制策略算法名      |
| 评分、优势、过滤、传输和训练结果 | 各自执行该操作的 span                                    | 保留逐 Export／Update 结果，后续按对象 ID 关联，不复制到 Agent span                    |

生成批次以输入任务为成员，训练批次以实际接纳的训练样本为成员，不能直接沿用生成批次的 ID 或数量。Actor 和 Critic 分别记录自身实际接纳的批次；仅在样本身份和成员集合完全一致时才可使用相同批次 ID。批次规模描述本阶段工作量，不能将各阶段的 size 相加作为全局样本总数。

Run／Step／Iteration 是查询和关联所需的上下文，仍按各表条件记录；模型、后端和版本描述本阶段实际执行对象，也可能因阶段不同而不同。各 span 的属性表列出其完整属性要求；未列出的其他阶段属性不复制，parent 或 Link 不会自动继承属性。轨迹流转使用逐 Export ID 关联。

**Trace 与关联规则：** 同一 iteration 的核心阶段在条件允许时复用 traceId，但不强制共用 trace；不同阶段之间不建立父子关系；同一阶段的批次与明细按第 2.1 节建立父子关系。通过业务 ID 关联查询，通过 Span Link 表达实际依赖。

1. **Trace 复用。** 第 3 章同一 iteration 内的核心阶段，已有合适的共同上游 Context 时应复用同一个 traceId，各次操作使用不同的 spanId，作为该共同上游下的同级 span，阶段之间不互为父子。没有这样的 Context 时，各批次阶段可以创建独立 trace 的根 span；本规范不为复用 traceId 新增统一的 Run／Step 父 span。按照 [OTel Span Creation](https://opentelemetry.io/docs/specs/otel/trace/api/#span-creation)，同一 trace 中的 spans 应属于同一棵调用树，不能将多个无 parent 的根 span 手动设置为相同 traceId。第 4 章的 Agent 调用仍保持独立 trace；同一次 Agent 调用的暂停恢复沿用原 span 和 traceId。
2. **业务聚合。** 无论是否复用 traceId，都通过 `rl.run.id` 聚合运行，通过 `(rl.run.id, rl.step.id)` 关联查询同一 Step 的各阶段和轨迹；查询必须支持一个 Step 对应多条 trace，不能依赖 traceId 完成整个 Step 的聚合。明确唯一 Step 归属时必须填写 `rl.step.id`；未知、未分配或跨 Step 时省略单值，按已有的真实业务归属明细关联，不能任选一个 Step。相同 Run／Step 只表示业务归属，不自动构成依赖关系。
3. **依赖关联。** 无论同 trace 还是跨 trace，有实际调用触发、数据消费或参数版本来源关系时，由被触发方／消费方的 span Link 到直接来源 span 的真实 SpanContext；阶段之间不通过 parent 表达该依赖。例如，训练 Link 到实际消费的优势计算／数据交付，权重加载 Link 到对应权重发布，Agent Link 到实际接纳或恢复它的 rollout。多来源操作关联各实际直接来源，并按 `(traceId, spanId)` 去重；不因同轮、时间相近或仅先后执行而建立 Link，不要求补齐所有间接祖先。
4. **上下文传递。** 跨进程、线程、队列及权重交付时，按实际关系传递 Context：符合第 1 条的共同上游 Context 可用于创建同级阶段、复用 traceId；直接来源的阶段 SpanContext 则用于 Link，不作为接收阶段 span 的 parent。创建时已知的依赖直接加入 Links，执行中确认的依赖在 span 结束前追加；不修改已经结束的来源 span。来源未被采集或上下文确实不可取得时，保留已有业务 ID 和明细，不根据 Run／Step／Export ID 生成虚假的 traceId 或 spanId。
5. **后端工作的 RL 封装。** SGLang、Megatron 的后端工作必须归属于负责该次操作的 RL 框架 span，由该 span 记录框架侧调用的完整起止与结果。直接复用本规范已有的阶段 span，不为每个后端内部 span 另建 RL 包装 span。例如，策略训练归属于 `rl.actor.train`，价值训练归属于 `rl.critic.train`，独立前向归属于对应的 forward span；普通生成归属于 `rl.rollout.sample`，其 parent 为首次接纳的 `rl.rollout.generate`；Agent 调用中的生成归属于 `rl.agent.run`，再由 Agent Link 到对应 sample。
6. **后端 trace 隔离与关联。** 上述封装表示业务归属，不建立 RL 与后端 span 的父子关系。后端 trace 必须独立于对应 RL trace，不继承 RL span 的 traceId，也不以 RL span 为 parent。后端 trace 的根 span 或对应调用的子树入口 span，通过 Span Link 指向负责该次操作的 RL span 的真实 SpanContext；传入的 RL Context 仅作为 Link 来源。SGLang、Megatron 内部已有的 span 名称、语义和真实父子关系继续沿用，同一后端子树中的后代通过入口 Link 追溯 RL 归属，无需逐个复制相同 Link。一个 RL span 可以关联多个后端 trace；共享后端工作涉及多个 RL 调用时，关联实际相关的 RL spans，不任选一个归属。
7. **规模与耗时。** 不将不同请求、rank 或更新已有的后端 traces 强行合并到一个 RL trace；通过上述隔离避免一次迭代的后端细节集中到同一条 trace。该规则控制单条 trace 的规模，不减少后端实际产生的 span 总量。各阶段按自己的真实起止时间结束；RL 阶段与后端操作，以及嵌套或并行阶段之间的耗时重叠，不得重复相加作为 Step 总耗时。

结构化数据直接保存为 span 的 string 类型 JSON 属性，包括题目、response、评分、优势、过滤、传输及训练结果。0.10.0 新采集不再输出对应的 `_ref`；完整内容随 OTLP JSONL 保存，不依赖附件或旧详情接口。禁止按长度截断，也不使用随机采样结果冒充全量数据。读取工具可以兼容旧版 `_ref`，但不能将旧引用当作新数据已内联的证据。

结构化内容中的 ID 为 string，计数为非负 int，结果数值为 double，布尔值为 boolean，`start_time`／`end_time` 为 Unix 秒 double。未知时间边界省略，后端原始字段保留原单位。普通 SDK 通常在 span 结束后导出；运行中展示需要同一 span 身份的状态／数据快照，终态记录去重合并，不增加业务 span 类型。

## 2.1 批次与逐样本明细

每个实际批次操作生成一个总 span，并为实际处理的每个成员生成明细 span。总 span 保存该操作的 `rl.batch.id`、实际输入数 `rl.batch.size` 和完整批次结果；明细直接保存所属成员的结果。生成批次按输入 Sample 计数，评分等下游批次按实际输入 Export 计数；不能假定两者总数相同。

| 批次 span                                                                             | 明细 span                     | 明细单位                                                         |
| ------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------- |
| `rl.rollout.generate`                                                                 | `rl.rollout.sample`           | 一条题目的一次完整生成，包括 Agent／工具交互；不包含独立评分     |
| `rl.reward.batch`                                                                     | `rl.reward.score`             | 一条 Export 的完整评分方案                                       |
| `rl.reference.forward`、`rl.teacher.forward`、`rl.critic.forward`、`rl.actor.forward` | 对应名称加 `.sample`          | 一条 Export 的独立前向结果                                       |
| `rl.advantage.compute`                                                                | `rl.advantage.compute.sample` | 一条 Export 的优势结果                                           |
| `rl.data.filter`                                                                      | `rl.data.filter.sample`       | 一条 Export 的 keep/drop 决定                                    |
| `rl.data.transfer`                                                                    | `rl.data.transfer.sample`     | 一条 Export 的交付结果                                           |
| `rl.actor.train`、`rl.critic.train`                                                   | 对应名称加 `.sample`          | 一条 Export 在一次 Update 中的实际消费；多个 Update 保留多条明细 |

明细与总 span 共用 traceId，parentSpanId 指向真实总 span。不同阶段仍通过 Link 关联：例如 `rl.reward.score` 的 parent 是 `rl.reward.batch`，其 Link 指向实际消费的 `rl.rollout.sample`。不能将 rollout 当作 reward 的 parent。SDK 创建明细时显式使用真实父 Context，不手动拼接 traceId。

**完整调用与窗口：** Sample 暂停／可续跑中断不结束 span，恢复沿用同一 sample.id、spanId、traceId 和首次 parent；后续窗口通过 Link 和 `rl.sample.memberships` 记录实际归属，不重新挂 parent。一个完整 Sample 可以晚于首次接纳窗口结束。Agentic 批次按 Shard 接纳的一整个 Group 生命周期定义：开始启动其 Sessions，直到所有已启动 Sessions 终态或启动失败清理结束；预取、暂停、恢复不新建批次。`rl.agent.run` 保持独立 trace，并 Link 到对应 sample。

**时间真实性：**

- `rl.sample.timing=operation`：存在可独立观测的成员调用，使用它的真实起止。普通生成和逐条评分属于此类。
- `rl.sample.timing=result`：后端一次处理整批／packed tensor，无法取得独立成员起止时，在埋点实际确认该成员结果的时刻生成零时长明细 span。其 start=end，不表示计算耗时为零。确认可能晚于批次原生结束；不得拉长批次 span 来包含该时刻。
- 不复制整个批次的时长到每个成员，不均分批次耗时，不因拥有结果数组就伪造独立执行。父、子或并行 span 的耗时不能相加作为阶段耗时。

所有明细为 `INTERNAL`，遵循第 2 章的状态规则；正常零分、drop、skipped 仍为 `UNSET`。未启动的成员不伪造执行 span；失败且已经启动但没有输出的成员必须保留。

**公共属性（补充各阶段属性表）：**

| Key                     | Stability   | Requirement Level                                  | Value Type | Description                                                |
| ----------------------- | ----------- | -------------------------------------------------- | ---------- | ---------------------------------------------------------- |
| `rl.batch.id`           | Development | Required：批次及其明细                             | string     | 当前阶段的实际输入集合身份；明细填真实 parent 的批次身份   |
| `rl.batch.size`         | Development | Required：批次输入确定时                           | int        | 实际输入成员数，不按 rank 重复，不记在明细上               |
| `rl.sample.id`          | Development | Required：rollout 明细；其他明细可取得时           | string     | 一次输入采样调用身份；同题重复采样不同，续跑保持不变       |
| `rl.export.id`          | Development | Required：下游成员使用 Export 身份时               | string     | 该结果所属的真实 Export ID，不以样本下标冒充               |
| `rl.sample.index`       | Development | Conditionally Required：框架只传递样本下标时       | string     | 框架实际输入样本下标，与 run/step 一起关联；不是 Export ID |
| `rl.sample.timing`      | Development | Required：明细                                     | string     | operation 或 result，按上述时间口径解释                    |
| `rl.sample.memberships` | Development | Conditionally Required：完整调用涉及多个窗口／Step | string     | JSON 数组，保存实际 batch_id、step_id，不覆盖历史          |

异步独立前向可按实际 DP 分区调用生成独立批次，`rl.partition.dp_rank` 记录整数分区编号；每个批次只计本分区真实成员，TP/PP/CP 不重复创建同一成员。训练 RPC 内可通过已有返回路径合并各 rank 的事实。

前向批次和明细的 `rl.forward.results` 为 JSON 数组，保存真实 `export_id` 或 `sample_index` 及结果状态；未独立采集的 token 概率不伪造。训练明细的 `rl.train.results.uses` 保存本条消费及已确认的 `update_status`。

下游明细保留 run.id、可确定的 step.id/iteration.index 及该阶段必要属性。普通训练仅有 sample_indices 时使用 `rl.sample.index`，rollout／评分同时保留同一个实际 index；不能凭空生成 Export 映射。数据不存在时不生成占位结果，采集不完整需明确诊断。

## 3 核心阶段 span

### Span: `rl.rollout.generate`

**Status:** Development

该 span 表示一个有界输入批次或采集窗口的轨迹生成操作，包含模型调用、Agent／工具／环境交互及生成结果收集。

从生成职责接纳输入开始，到本窗口关闭并交付已有结果，或失败／取消处理结束。独立 reward、训练准备和后续训练不属于此边界。

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

普通部分 rollout 窗口可以在样本暂停时结束，完整 sample 保持未结束；窗口 Status 按交付承诺判断。Agentic 总批次覆盖完整 Group，跨轮保持不变。每次实际 Agent 调用使用第 4 章的独立 trace，并 Link 到对应 `rl.rollout.sample`。

### Span: `rl.rollout.sample`

**Status:** Development

该 span 表示每条题目的一次完整 rollout。普通模式包含该输入的全部生成调用；Agentic 模式对应一个 Session，包含所有模型／工具交互以及输出构建。一个 Sample 可以产出多个 Export，不按 Export 数重复创建该 span。

从首次接纳此输入开始，到生成结果完成或不可恢复的失败／取消。可恢复中断和暂停不结束。已有完成结果被再次读取时不新建 span。parent 固定为首次接纳的 `rl.rollout.generate`，恢复窗口按第 2.1 节关联。

**Span kind:** `INTERNAL`。**Span name:** 固定为 `rl.rollout.sample`。

**Span status:** 按第 2 章；已开始但失败且没有 Export 时也必须结束并记录错误。

**Requirement level:** `Required`；实际生成一个输入时采集。

除第 2.1 节公共属性，还记录：

| Key                                                    | Stability   | Requirement Level                    | Value Type | Description                                                                            |
| ------------------------------------------------------ | ----------- | ------------------------------------ | ---------- | -------------------------------------------------------------------------------------- |
| `rl.run.id`                                            | Development | Required                             | string     | 本次运行身份                                                                           |
| `rl.purpose`                                           | Development | Required                             | string     | training 或 evaluation                                                                 |
| `rl.rollout.mode`                                      | Development | Required                             | string     | single_turn、multi_turn 或 agentic                                                     |
| `rl.session.id`                                        | Development | Conditionally Required：Agentic      | string     | 对应完整 Session 身份                                                                  |
| `rl.sample.input`                                      | Development | Required：输入可取得时               | string     | 完整原始题目／messages 的 JSON；保持文本或消息结构，不填格式化后的猜测                 |
| `rl.sample.response`                                   | Development | Conditionally Required：已有实际输出 | string     | JSON 数组，每项保存 export_id、可取得的 name 及 text 或 messages；多 Export 不任选一条 |
| `rl.model.name`、`rl.backend.name`、`rl.model.version` | Development | Conditionally Required：唯一且已知   | string     | 本次真实执行的模型、后端和参数快照                                                     |

普通 rollout 的 response 保存实际生成文本和框架状态 status；取消／失败时保留已有部分文本，`rl.sample.cancelled=true` 表示已确认丢弃，不把部分文本冒充完整答案。Agentic response 由各 Export 的状态树分支取得最后的 assistant 消息，不把混合工具观察的训练 continuation 冒充最终回答。完整轨迹继续记录在 `rl.agent.run`。失败／无输出时省略 response，不用空字符串表示未知。

### Span: `rl.reward.batch`

**Status:** Development

该 span 是一批数据评分的总览，包含实际逐条 `rl.reward.score` 子 span。输入流式到达时，批次按已接纳的 rollout 窗口／明确的评分调用集合界定，使用新的评分 batch.id；batch.size 只计实际启动评分的输入，读取已有 reward 不计入。

从首个评分开始，到输入窗口封口且所有已启动评分结束。末尾无评分的生成／后处理不计入原生结束时间；一个子评分提前失败不能关闭仍运行的兄弟评分。真正的批量 scorer 使用一次批次 span，成员结果按 result 明细记录。

**Span kind:** `INTERNAL`。**Span name:** 固定为 `rl.reward.batch`。

**Span status:** 未处理的评分失败为 ERROR，正常零分／负分保持 UNSET。

**Requirement level:** `Required`；有实际评分时采集。

属性为第 2.1 节批次公共属性和 `rl.reward.score` 中的 run、purpose、step、评分方案、模型属性。混合评分路由使用完整方案名；批次 `rl.reward.results` 保存完整逐成员结果，子 span 保存自己对应的结果，不以平均值替代明细。

### Span: `rl.reward.score`

**Status:** Development

该 span 表示对一条已有轨迹执行完整原始评分，包括规则、验证器、奖励模型或模型裁判。parent 为真实 `rl.reward.batch`；独立调用使用 operation 时间，批量 scorer 的成员结果使用 result 时间。

从评价逻辑开始，到分数及有效性结果可供下游使用。读取已有 reward 不代表再次评分，不生成虚假的评分耗时。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.reward.score`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                                  | Value Type | Description                                                        | Example Values                                                                |
| -------------------- | ----------- | -------------------------------------------------- | ---------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| `rl.run.id`          | Development | Required                                           | string     | 本次实际运行的唯一身份。                                           | `run-a7`                                                                      |
| `rl.purpose`         | Development | Required                                           | string     | 操作用途：training 或 evaluation。                                 | `training`                                                                    |
| `rl.reward.name`     | Development | Required                                           | string     | 完整评分方案的稳定名称。                                           | `math_verifier`                                                               |
| `rl.reward.type`     | Development | Required                                           | string     | rule、verifier、model、judge、environment 或 hybrid。              | `verifier`                                                                    |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step          | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                         | `learner:80`                                                                  |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次           | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                     | `80`                                                                          |
| `rl.model.name`      | Development | Conditionally Required：采用模型评分且使用单个模型 | string     | 当前操作使用的模型名称，不用算法名称代替。                         | `reward-model`                                                                |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一     | string     | 实际模型执行后端；Ray 不属于模型执行后端。                         | `megatron`                                                                    |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照       | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。 | `snapshot-48`                                                                 |
| `rl.reward.results`  | Development | Required：对应结果可取得时                         | string     | 逐 Export 的评分结果。\[1\] 值为完整 JSON 字符串。                 | `[{"export_id":"export-1","source":"verifier","status":"valid","value":2.0}]` |

**\[1\] `rl.reward.results`:** 每项包含 export_id、source、status（valid／invalid／error）、标量或复合 value；复合值明确标量选择口径，失败／无效时记录 reason。不能只保存批次平均值。零分、负分或答案错误可以是成功评分；多个子评分组成方案时，核心 span 表示完整方案。

### Span: `rl.reference.forward`

**Status:** Development

该 span 表示参考策略对已有 token 执行无梯度前向，提供 reference logprob，用于 KL 等约束。

从开始独立参考前向，到参考概率输出可用。临时关闭 adapter 或切换参考权重不改变此逻辑职责。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.reference.forward`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                              | Value Type | Description                                                        | Example Values |
| -------------------- | ----------- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------ | -------------- |
| `rl.run.id`          | Development | Required                                       | string     | 本次实际运行的唯一身份。                                           | `run-a7`       |
| `rl.model.name`      | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                         | `reference-7b` |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                         | `learner:80`   |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                     | `80`           |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                         | `megatron`     |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照   | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。 | `snapshot-48`  |

### Span: `rl.teacher.forward`

**Status:** Development

该 span 表示教师对已有学生轨迹执行评分前向，提供在线蒸馏的概率目标。

从开始教师评分前向，到所需概率目标可用。教师另行生成答案时属于生成调用，不能记成固定轨迹 forward。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.teacher.forward`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                              | Value Type | Description                                                                 | Example Values      |
| -------------------- | ----------- | ---------------------------------------------- | ---------- | --------------------------------------------------------------------------- | ------------------- |
| `rl.run.id`          | Development | Required                                       | string     | 本次实际运行的唯一身份。                                                    | `run-a7`            |
| `rl.model.name`      | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                                  | `teacher-32b`       |
| `rl.forward.output`  | Development | Required                                       | string     | 教师目标类型：sampled_token_logprob、topk_distribution、full_distribution。 | `topk_distribution` |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                                  | `learner:80`        |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                              | `80`                |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                                  | `megatron`          |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照   | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。          | `snapshot-48`       |

教师概率目标与 reference 约束、原始 reward 是不同职责，不能因接口复用而混同。

### Span: `rl.critic.forward`

**Status:** Development

该 span 表示更新前的价值模型对已有轨迹估值，为优势和 returns 构造提供旧 values。

从开始独立价值预测，到 values 可用。价值模型训练内部的带梯度前向属于 critic.train。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.critic.forward`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                              | Value Type | Description                                                        | Example Values |
| -------------------- | ----------- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------ | -------------- |
| `rl.run.id`          | Development | Required                                       | string     | 本次实际运行的唯一身份。                                           | `run-a7`       |
| `rl.model.name`      | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                         | `critic-7b`    |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                         | `learner:80`   |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                     | `80`           |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                         | `megatron`     |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照   | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。 | `snapshot-48`  |

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

**\[1\] `rl.advantage.results`:** 每项包含 export_id、mean、token_count。均值按该阶段有效位置计算，无有效位置时省略 mean；需要展示原始分项时保留 values 及其粒度。实际训练 mask 发生变化时，训练结果使用最终 mask 对应的摘要。

### Span: `rl.actor.forward`

**Status:** Development

该 span 表示策略对已有轨迹独立执行无梯度前向，准备固定概率基准或学生蒸馏概率。

从开始独立策略评分，到所需概率输出可用。训练内部带梯度前向不属于此操作；直接复用 rollout logprob 时不创建此 span。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.actor.forward`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Attributes:**

| Key                  | Stability   | Requirement Level                              | Value Type | Description                                                        | Example Values |
| -------------------- | ----------- | ---------------------------------------------- | ---------- | ------------------------------------------------------------------ | -------------- |
| `rl.run.id`          | Development | Required                                       | string     | 本次实际运行的唯一身份。                                           | `run-a7`       |
| `rl.model.name`      | Development | Required                                       | string     | 当前操作使用的模型名称，不用算法名称代替。                         | `policy-7b`    |
| `rl.forward.purpose` | Development | Required                                       | string     | 独立策略评分的用途：old_policy 或 distillation。                   | `old_policy`   |
| `rl.step.id`         | Development | Conditionally Required：明确唯一所属 Step      | string     | Run 内的 Step 身份；跨 Step 时不任选一个。                         | `learner:80`   |
| `rl.iteration.index` | Development | Conditionally Required：明确唯一外层轮次       | int        | 原框架的外层 RL 执行轮次，不是优化器更新次数。                     | `80`           |
| `rl.backend.name`    | Development | Conditionally Required：模型执行后端已知且唯一 | string     | 实际模型执行后端；Ray 不属于模型执行后端。                         | `megatron`     |
| `rl.model.version`   | Development | Conditionally Required：使用唯一已知参数快照   | string     | 实际使用／交付的参数版本，不填期望版本；混合版本在轨迹中逐段记录。 | `snapshot-48`  |

### Span: `rl.actor.train`

**Status:** Development

该 span 表示对有界批次或流式输入窗口执行策略训练，包含带梯度前向、loss、反向、梯度累积和优化器更新决策。

从接纳本次训练输入／窗口并开始训练，到该窗口更新流程完成，或失败／取消处理结束。流式输入等待计入端到端时间；权重发布、checkpoint 和训练后 offload 不在此边界内。

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

**\[1\] `rl.train.results`:** 包含 updates 和 uses 两个列表。updates 保存 update_id、已确认的 status（committed／skipped／failed）、reason，以及可获得的后端 span_context、source_name、start_time、end_time。uses 保存每个 `(export_id, update_id)` 的 used、最终 effective_tokens 和可取得的 advantage_mean，保留多 epoch 的多次消费关系。

更新是否提交取自真实决策，不能由训练函数正常返回推断。used=true、有效 token 大于 0、参数更新已提交是不同事实。时间只引用已有后端记录，不新增 optimizer span，也不能把 train_batch 整段冒充纯 optimizer 耗时。

### Span: `rl.critic.train`

**Status:** Development

该 span 表示使用 returns 等目标训练价值模型，包含价值模型的带梯度前向、loss、反向和更新决策。

从接纳价值训练输入／窗口并开始训练，到本窗口价值更新流程完成，或失败／取消处理结束。独立旧 values 预测属于 critic.forward；流式输入等待计入端到端时间。

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

**\[1\] `rl.train.results`:** 包含 updates 和 uses 两个列表。updates 保存 update_id、已确认的 status（committed／skipped／failed）、reason，以及可获得的后端 span_context、source_name、start_time、end_time。uses 保存每个 `(export_id, update_id)` 的 used、最终 effective_tokens 和可取得的 advantage_mean，保留多 epoch 的多次消费关系。

更新是否提交取自真实决策，不能由训练函数正常返回推断。used=true、有效 token 大于 0、参数更新已提交是不同事实。时间只引用已有后端记录，不新增 optimizer span，也不能把 train_batch 整段冒充纯 optimizer 耗时。

### Span: `rl.actor.publish_weights`

**Status:** Development

该 span 表示将一份一致的策略快照交付给约定接收端，使其可以获取该版本。

从开始序列化、发布或传输，到源端交付完成：拉取式为快照已可读取，推送式为约定目标已确认接收。此边界不包含目标激活。

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

从目标开始获取或安装参数，到本次声明的目标范围确认生效。单个 rank 安装完成不能代替整个目标可用的确认。

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

**\[1\] `rl.filter.results`:** 每项包含 export_id、decision（keep／drop）；drop 时包含 reason。整组过滤也保留每个成员身份，不能只记录组级数量。

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

**\[1\] `rl.transfer.data`:** 保存 export_ids，以及已知的入 buffer、入队、发送开始和接收确认时间。单个成员的边界不同时保留其 Export ID。结束边界缺失时保持缺失，不能根据后续训练成功反推传输完成。

## 4 Agent 调用 span

### Span: `rl.agent.run`

**Status:** Development

该 span 表示一次 Agent 调用及其产出的轨迹。一个 Session 对应一个 Agent span，包含多轮模型调用、工具交互、状态树及多个 Export。失败且没有 Export 的调用也必须保留。

从 Agent 启动，到调用终态、Export 构建和结果交付结束。轨迹中的 execution 单独记录 Agent 实际执行起止，区分执行时间与导出／清理时间。

**Span kind:** `INTERNAL`。

**Span name:** 固定为 `rl.agent.run`。

**Span status:** 正常结束保持 `UNSET`；失败设置 `ERROR` 和 `status_message`，具体规则见第 2 章。

**Requirement level:** `Required`；实际执行该操作时采集。

**Span links:** 按第 2 章创建独立 trace 的根 span，不设置 parent；必须 Link 到对应 `rl.rollout.sample` 的真实 SpanContext；sample 是实际 Group 生成批次的子 span。暂停恢复保留同一个 Agent span 和 traceId，跨轮保留实际 Step 归属。预取阶段提前终止且从未分配的调用仍有 sample 和批次，必须标记 unassigned。

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
