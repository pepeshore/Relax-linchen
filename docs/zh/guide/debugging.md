# 调试指南

用于排查 Relax 训练中精度问题和隔离训练/推理组件的实用指南。

## 精度排查

### 1. 检查 Rollout 生成的 Response

在日志中搜索 `Finish rollout`，定位当前的 rollout ID 和生成的 response。首先判断 response 是否为正常的、连贯的回复。

- **如果第一步就出现了乱码/胡言乱语**，一般是由于 checkpoint 加载异常或模型转换异常。一个比较彻底的方法是在对应模型的 SGLang `load_weights` 实现中保存所有参数，查看与加载的 checkpoint 中是否一致。如果所有参数更新都正确还出现问题，有可能是 SGLang 里有一些特殊的 buffer 在 `release_memory_occupation` 阶段被释放了。如果是用 pretrain 模型进行的测试，可以换成同结构模型的 instruct 版本，查看这种乱码是不是 pretrain 模型特有的。

- **如果第一步回复正常，但后续 step 出现乱码**，说明训崩了。需要更细致地去检查每一步的 reward 计算是否符合预期、超参数配置等因素。

### 2. 检查 log_probs 和 ref_log_probs

查看第一步打印的 rollout stats，确认 `log_probs` 和 `ref_log_probs` 是否完全相等（即第一步 KL = 0），且值较小。

- **如果不是完全相等**，一般是 Transformer Engine 中的某些非确定性 kernel 导致的。例如，在某些版本的 TE 里，Megatron 需要 `--attention-backend flash` 来强制使用 Flash Attention，从而避免 Context Parallelism (CP) 下 fused attention 的数值不稳定。

- **如果数值较大**（例如 > 1），一般有两种可能：
  - 值非常大，应该是训练配置有问题。
  - 值只是比 SFT loss 的状态略大（例如 instruct 模型的 logprob 到了 0.8），有可能是数据不符合训练的 chat template，或者不符合冷启动的分布。

### 3. 检查第一步的 KL 和 grad_norm

在推一训一（`num_steps_per_rollout == 1`）的配置下，检查第一步的 KL 是否为 0，`grad_norm` 是否较小。

此阶段的问题基本上是 Megatron / Transformer Engine 相关的 bug。例如：

- MoE 模型需要开启 `--moe-permute-fusion`。

## SGLang 运行时崩溃

### 1. Stop string 在 agentic 高并发下触发 IMA

**症状**：agentic rollout 跑到中段（rollout 已经运行了几十秒到几分钟）时，SGLang scheduler 抛异常，紧接着 `SIGQUIT received. ... It usually means one child failed`，engine 被 kill，整个 rollout 停摆：

```
[TP0] Scheduler hit an exception: Traceback (most recent call last):
  ...
  File "sglang/srt/managers/scheduler_output_processor_mixin.py", line 371, in process_batch_result_decode
    next_token_ids = next_token_ids.tolist()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**根因**：OpenAI-compatible request 里的 `stop` 字符串由 SGLang detokenizer 子进程按 per-token 增量解码 + 子串匹配处理。agentic 场景下 100+ session 并发（每个 session 内又有多轮 chat completion），detokenizer 的单 token 工作量随 stop string 数量线性放大，容易 starve 掉它的 20s 心跳，GPU 侧 async kernel 就会以 IMA 的形式浮现。

**规避**：能用 `stop_token_ids` 就不要用 `stop`。`stop_token_ids` 走 GPU 侧 token id 匹配，完全绕过 detokenizer 子进程。

- 单 token 就能表示的终止符（如 `</tool_call>`、`<|im_end|>`）：查 tokenizer 拿到 id 后直接传 `stop_token_ids`。
- 多 token 序列（如 `</answer>` = `[510, 8944, 29]`）无法用 `stop_token_ids` 表达。这种情况优先用 chat-template 的 turn 终止符（`<|im_end|>`）作为停止条件，再在生成后 regex 提取答案，仍然可以避开 `stop`。

在 agent app 里通过 `extra_body` 传 `stop_token_ids`：

```python
extra_body = {"stop_token_ids": [248059, 248046]}  # </tool_call>, <|im_end|>
resp = await client.chat.completions.create(
    model=...,
    messages=messages,
    extra_body=extra_body,
    ...
)
```

具体示例可参考 `examples/deepeyes_v2_agentic/app/agent.py` 与 `examples/deepeyes_v2_agentic/app/deepeyes_v2_config.yaml` 的注释。

## 训练推理单独调试

Relax 支持将训练和推理组件分开独立运行，从而实现：

- 在调试推理部分时，只用少量卡就可以启动任务。
- 在调试训练部分时，可以保证模型输入固定，去除 rollout 的随机性。

### 可用的调试参数

以下 CLI 参数用于开启隔离调试：

| 参数 | 描述 |
|---|---|
| `--debug-rollout-only` | 仅初始化 SGLang（跳过 Megatron）。用于推理调试。 |
| `--debug-train-only` | 仅初始化 Megatron（跳过 SGLang）。用于训练调试。 |
| `--save-debug-rollout-data <path>` | 将 rollout 结果保存到指定路径，供后续回放使用。 |
| `--load-debug-rollout-data <path>` | 从指定路径加载 rollout 数据。自动设置 `--debug-train-only`。 |
| `--dump-details <dir>` | 保存训练的全部细节（自动开启 rollout 数据保存）。 |
| `--load-forge-rollout-data <path>` | 回放已保存的 rollout dump，**同时保持 SGLang 存活**（不会设置 `--debug-train-only`）。需配合 `--rollout-function-path relax.engine.rollout.forge_load.generate_rollout`。 |

### 工作流 1：仅调试推理

使用 `--debug-rollout-only` 跳过 Megatron 初始化。只会启动 SGLang 引擎，可以用更少的 GPU 来测试推理部分。

```bash
python3 relax/entrypoints/train.py \
    --debug-rollout-only \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 8 \
    # ... 其他 rollout 参数
```

可以配合 `--save-debug-rollout-data` 来保存 rollout 结果：

```bash
python3 relax/entrypoints/train.py \
    --debug-rollout-only \
    --save-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt \
    # ... 其他 rollout 参数
```

### 工作流 2：仅调试训练

使用 `--load-debug-rollout-data` 加载预先保存的 rollout 数据，仅运行训练流程。这会自动设置 `debug_train_only=True`，因此不会初始化 SGLang。

```bash
python3 relax/entrypoints/train.py \
    --load-debug-rollout-data /your/saved/debug/data_{rollout_id}.pt \
    # ... 其他训练参数
```

这种方式特别适用于：

- 在不等待 rollout 的情况下调优并行配置（TP、PP、EP、CP）。
- 用确定性输入来复现和修复训练相关的问题。
- 快速迭代 loss 计算或优化器的修改。

### 工作流 3：全量细节转储

使用 `--dump-details` 保存训练的全部细节供事后分析。设置后会自动启用：

- `--save-debug-rollout-data`，保存路径为 `<dir>/rollout_data/{rollout_id}.pt`
- `--save-debug-train-data`，保存路径为 `<dir>/train_data/{rollout_id}_{rank}.pt`

```bash
python3 relax/entrypoints/train.py \
    --dump-details /path/to/dump/dir \
    # ... 其他参数
```

::: tip 提示
`--dump-details` 在收集 bug 报告数据时也非常有用 — 它会捕获复现问题所需的所有信息。
:::

### 工作流 4：保持 SGLang 存活回放 Rollout（显存分析）

`--load-debug-rollout-data`（工作流 2）会完全跳过 SGLang，因此只能孤立地测量训练循环。若要分析**真实的 colocate 显存占用** —— 训练峰值、权重同步、以及 offload/onload 切换 —— 就需要 SGLang、router 和权重更新全程存活。`forge_load` 用回放已保存的 rollout dump 来替代真实生成，从而保持整条流水线存活，只省掉（缓慢的）真实生成。

先抓一份 dump（例如通过工作流 1 或 `--dump-details`），再回放：

```bash
python3 relax/entrypoints/train.py \
    --rollout-function-path relax.engine.rollout.forge_load.generate_rollout \
    --load-forge-rollout-data /path/to/rollout_data/data_{rollout_id}.pt \
    # ... 保持与产生该 dump 的那次运行相同的 模型 / colocate / batch 参数
```

`--load-forge-rollout-data` 支持两种写法，沿用 `--save-debug-rollout-data` 的 `format(rollout_id=...)` 约定：

- **字面路径（literal）** —— `--load-forge-rollout-data /path/to/rollout_data/0.pt`
  不含 `{rollout_id}` 占位符：**每个** rollout step 都复用同一个文件。适合用一份 dump 反复回放 `--num-rollout > 1` 的显存测试。
- **模板路径（template）** —— `--load-forge-rollout-data /path/to/rollout_data/data_{rollout_id}.pt`
  含 `{rollout_id}`：每个 step 加载**各自**的文件（`data_0.pt`、`data_1.pt` ……）。若某个 step 的文件缺失，训练路径会回退到 `0` 那份 dump。

::: warning 注意
batch 相关参数（`--global-batch-size`、`--n-samples-per-prompt`、并行度）要与产生 dump 的那次运行保持一致 —— dump 必须恰好包含一个 step 的样本量。**不要**与 `--debug-rollout-only` 或 `--load-debug-rollout-data` 同时使用，二者都会破坏其目的（不训练，或跳过 SGLang）。`forge_load` 不会触发解码期显存，因此不能替代对生成阶段峰值的测量。
:::

## 本地 RL Trace

通过 `relax.entrypoints.train` 启动训练时，默认将 RL 阶段 trace 写入磁盘。每次启动都会生成新的 `rl.run.id`，从 checkpoint 恢复也不例外。文件默认位于 `<save>/otel-traces/<run_id>/`；没有设置 `--save` 时，使用 driver 工作目录下的 `otel-traces`。每个进程独立写入一个 OTLP JSONL 文件，每行是一次 OTLP trace 导出请求，分析时逐行合并读取。

提交任务前设置 `RELAX_OTEL_DIR` 可更改输出根目录，设置 `RELAX_OTEL_ENABLED=0` 可关闭 RL trace，`OTEL_SDK_DISABLED=true` 也会关闭采集。若需要在一台机器上读取所有 worker 的数据，请使用共享持久化目录；使用节点本地磁盘时，需要分别收集各 worker 的文件。采集不依赖 Collector 地址。

内置普通 rollout 和 Agentic rollout 会通过 W3C headers 将对应 RL span 传给 SGLang：普通请求的 parent 是 `rl.rollout.sample`，Agentic 请求使用该回调中有效、同 trace 的父上下文，缺失、无效或跨 trace 时使用当前 `rl.agent.run`。SGLang 请求继承相同 trace ID，其内部父子树保持不变。使用 `--sglang-enable-trace --sglang-otlp-traces-endpoint "$SGLANG_TRACE_ENDPOINT"` 启用后端采集；该地址必须能被各推理节点访问。后端仍单独导出，不会自动写入上述 RL 文件。

仓库提供一个用于实验验证的 OTLP/gRPC 接收器，在装有 SGLang tracing 依赖的环境中运行：

```bash
PYTHONPATH=. python scripts/tools/collect_otlp_traces.py \
  --listen "$OTLP_LISTEN" \
  --output "$SAVE_DIR/$EXP_NAME/backend-traces/sglang.jsonl"
```

`OTLP_LISTEN` 为本机绑定的 `host:port`，`SGLANG_TRACE_ENDPOINT` 为推理节点可访问的对应地址。接收器每收到一批数据就追加一行 OTLP JSONL，关闭前等待训练请求完成及 SGLang 导出。合并分析 RL 与后端文件时，按 trace/span/parent ID 还原调用树；RL 业务属性可沿父链查询，不会自动复制到 SGLang span。

`rl.agent.run` 是 Relax 的 Session span。外部 Agent 应用及工具不会自动产生自己的 OTel spans，也不会自动接入这个父链。自定义 rollout 如果自行使用 HTTP 客户端，可在真实生成调用内用 `relax.utils.rl_tracing.inference_headers` 合并出站 headers；绕过内置入口的请求需要自行接入。

示例 `examples/otel_tracing/agent.py` 在进入 Session 时创建 `agent.invoke`，父节点是 Relax 传入的 `rl.agent.run`。每次模型请求创建一个 `llm.call`，并将它的上下文注入回调。两轮解题默认只调用模型。扩展示例时，用 `await call_tool(工具名, 函数, *args, **kwargs)` 执行同步或异步工具，`tool.call` 会覆盖从调用到返回或抛错的时间。示例使用 `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` 导出应用 spans，Agentic smoke 脚本会将它设为 SGLang 使用的接收器。

采集高吞吐、多轮推理时，导出队列不足可能导致 spans 丢失，即使请求入口关联正确也会出现缺失的父节点。上述 smoke 脚本启用 SGLang tracing 时默认使用 `OTEL_BSP_MAX_QUEUE_SIZE=65536` 和 `SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE=512`，可通过环境变量覆盖。接收器建议先写节点本地磁盘，训练及导出结束后再复制到持久化存储；应校验完整父链，不能仅检查请求入口数量。

通过 `rl.run.id` 和 `rl.step.id` 查询同一步的各条 trace。`rl.iteration.index` 保留从 0 开始的 rollout 轮次。每个批次记录自身 `rl.batch.id` 和实际输入数，真实成员操作 span 与批次共用 trace 并指定真实 parent：`rl.rollout.sample` 记录一条输入的完整 rollout，`rl.reward.score` 归属于 `rl.reward.batch`。数据来源使用 Links。每次普通或 Agentic rollout 调用产生一个 `rl.rollout.generate`，直接包含该调用的样本和实际发起的各个 `rl.reward.batch`，其耗时包含函数返回前的等待。独立发起的评分保留根 span。不同 prompt group 的样本通过 `rl.group.id` 区分。`rl.agent.run` 是本 Session 样本的子 span。Agent 可提取子进程环境中的 `RELAX_TRACEPARENT`／`RELAX_TRACESTATE`，在发起模型请求时注入当前调用的 W3C 上下文。

结构化结果、题目和 response 直接以完整 JSON 字符串保存在 span 属性中，新 trace 不依赖 `_ref` 附件。仅真实的逐样本 rollout 和评分调用创建子 span，标记 `rl.sample.timing=operation`。批量评分、前向、优势、过滤、传输和训练将成员结果保存在批次 span 上，不生成零时长结果 span。OPD 单样本前向只保留一个真实请求 span。训练 uses 通过 update_id 关联 updates 中的更新结果。旧版 0.10 的 result span 可兼容读取，但应与执行时间线分开展示。父子 span 的耗时也不能相加。Agentic 运行还会启用已有 observation 记录，保存 Session 的请求、尝试、状态及 Export，并通过 Export ID 关联实际训练消费。无法观测的外部工具时间保持未知。这些记录包含模型输入输出，应根据训练数据设置目录权限和保留期限。

通过训练 RPC 汇总的 Megatron 阶段使用 `rl.timing.scope=representative_rank`：主时间线固定来自 DP0、CP0、TP0、最后一级 PP，业务结果仍汇总所有实际参与的 rank。`rl.timing.ranks` 保存每个来源的原始起止时间及主机／进程身份，其他 rank 不再拉长主 span。advantage 排除未执行计算的 PP stage。指定来源缺失时省略该 span 并记录诊断，不临时换另一个时钟。

RL 耗时统一标记 `rl.timing.clock=host_wall`，包含调用内部已有的等待，不表示 GPU kernel 时间。train 从实际训练函数入口开始，不包含此前 RPC 准备／预取、之后的 checkpoint 和 offload。结果整理与 trace 导出不会延长已完成操作；自定义批量评分的实际后处理仍计入，即使结果值未改变。可恢复 Sample 保持原起点，直到真正终态才结束。权重 span 使用 `source_sync`，表示源端从准备到交付／生效确认，包含暂停、转换和等待，不是目标端独立安装时间。采集不会新增设备同步或分布式 barrier。

阶段结束后即追加写入文件。进程在阶段执行中被强制终止时，该阶段可能没有结束记录。写入失败会记日志，训练继续运行；rank 数据不完整时，通过 `relax.observation.partial` 保存诊断信息，不冒充完整训练结果。采集不会增加分布式通信，GPU 上的简要统计通过异步复制交给训练线程之外的读者处理。

训练结束后，可检查文件格式、内联结果、父子关系和阶段覆盖：

```bash
python scripts/tools/validate_rl_traces.py /path/to/otel-traces/RUN_ID --min-train-steps 2
```

如果存在失败 span、rank 数据不完整、结果格式错误、明细缺少父 span 或训练轮次不足，检查会失败。工具汇总实际记录的阶段和优化器决策，不代表本次未运行的算法或路径也得到验证。
