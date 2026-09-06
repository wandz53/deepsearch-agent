# DeepSearch Agent

基于 **DeepAgents / LangGraph / LangChain** 的多智能体研究系统二次开发项目。

> 本仓库基于开源项目 [`didilili/deepsearch-agents`](https://github.com/didilili/deepsearch-agents) 进行学习与改造。  
> 当前个人重点聚焦于 **Agent 编排、状态管理、Backend / Memory、Middleware、异步上下文隔离与执行监控**，并将在此基础上继续加入 Reviewer Agent 与经验沉淀机制。

---

## 项目定位

当前项目采用 **Orchestrator-Worker** 主从式多智能体架构：

```text
User
  |
  v
Main Agent / Orchestrator
  |
  +--> SubAgent A
  +--> SubAgent B
  +--> SubAgent C
  |
  v
Result
```

主 Agent 负责理解任务、选择合适的子 Agent、分发子任务并汇总结果；子 Agent 负责执行相对专业化的任务。

当前个人学习与改造范围主要围绕：

- Tool Calling 与 Agent 执行闭环
- 主 Agent / SubAgent 任务路由
- `astream()` 与异步并发执行
- LangGraph / LangChain Agent 接入 DeepAgents
- Human-in-the-loop 与中断恢复
- Checkpointer 与 `thread_id`
- FilesystemBackend / StoreBackend / CompositeBackend
- Middleware、Model/Tool Call Limit、`wrap_tool_call`
- Skills 按需加载
- `ContextVar` 请求上下文隔离
- Monitor / ConnectionManager / WebSocket 执行事件推送
- 模型初始化与 Prompt 配置管理

---

## 当前核心能力

### 1. 多智能体任务编排

通过主 Agent 统一调度专业 SubAgent，并利用子智能体 `description` 进行能力路由。

结合 `task` 调用完成子任务分派，并通过流式事件观察模型决策、工具调用与子智能体执行过程。

### 2. 异步与流式执行

使用 `astream()`、`asyncio` 与 `asyncio.gather()` 处理长任务中的异步执行与并发调用，减少 I/O 等待场景下的串行阻塞。

### 3. Human-in-the-loop 与任务恢复

通过 `interrupt_on`、Checkpointer、`thread_id` 与 `Command(resume=...)`，对高风险 Tool 调用增加人工审批，并支持 `approve / edit / reject` 后从中断位置继续执行。

### 4. Backend 与长期记忆

将 Agent 执行状态和长期数据分开处理：

```text
Checkpointer
  -> 保存当前任务执行状态

Backend
  -> 管理 Agent 工作区文件和跨任务长期数据
```

当前涉及 `FilesystemBackend`、`StoreBackend` 与 `CompositeBackend`，用于工作区文件管理、跨 `thread_id` 数据共享和不同路径的存储路由。

### 5. Middleware 执行治理

在 Tool 调用链前后加入 Middleware，用于：

- 模型调用次数限制
- Tool 调用次数限制
- 日志与耗时记录
- 参数校验
- 异常处理
- 后续结果处理与脱敏扩展

核心执行结构：

```text
before handler(request)
        |
        v
 handler(request)
        |
        v
after handler(request)
```

### 6. 并发任务上下文隔离

使用 `ContextVar` 保存当前任务的 `thread_id` 与 `session_dir`，避免 FastAPI / asyncio 并发任务之间出现上下文、文件目录或监控消息串台。

### 7. Agent 执行监控

通过 `ConnectionManager` 维护：

```text
thread_id -> WebSocket
```

并由 Monitor 将工具调用、子智能体调用、任务结果和异常等事件推送到对应前端连接。

在后台 Agent 与 WebSocket 所属事件循环不一致时，通过 `asyncio.run_coroutine_threadsafe()` 将消息安全投递回正确的 event loop。

---

## 下一阶段改造

当前计划在现有 Orchestrator-Worker 架构上增加一个小型的 **Execute → Review → Retry → Experience** 闭环：

```text
                 Main Agent
                     |
                     v
              Research Agent
                     |
                     v
              Reviewer Agent
                 /       \
          not pass       pass
             |             |
             v             v
        retry / revise   finish
             |
             v
        StoreBackend
       experience memory
```

计划包含三个改造点：

### Reviewer Agent

将“任务执行”和“结果审核”拆开，由独立 Reviewer Agent 检查结果完整性、逻辑一致性、证据缺失和是否需要重新执行。

### Review-Retry Loop

当 Reviewer 判定结果不满足要求时，由主 Agent 触发 Research Agent 再次执行或修正，而不是直接结束任务。

### Experience Memory

把经过验证的有效方法、工具组合、失败边界等内容写入 `StoreBackend`，让后续新的 `thread_id` 可以复用历史经验。

---

## 项目结构

```text
deepsearch-agents/
├── app/
│   ├── agent/
│   │   ├── subagents/
│   │   ├── llm.py
│   │   ├── main_agent.py
│   │   └── prompts.py
│   ├── api/
│   │   ├── context.py
│   │   ├── monitor.py
│   │   └── server.py
│   ├── prompt/
│   │   └── prompts.yml
│   ├── tools/
│   └── utils/
├── frontend/
├── docs/
├── examples/
├── docker/
├── .env.example
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 技术栈

| 类型 | 技术 |
| --- | --- |
| Agent | DeepAgents / LangChain / LangGraph |
| Language | Python |
| Async | asyncio |
| Backend | FastAPI |
| Realtime | WebSocket |
| Storage | FilesystemBackend / StoreBackend / CompositeBackend |
| Context | ContextVar / thread_id / session_dir |
| Config | `.env` / YAML |
| Document | Markdown / ReportLab |

---

## 当前说明

本仓库目前保留了上游开源项目的完整基础代码，用于后续二次开发。

**个人简历与面试当前只使用自己已经学习、理解和实际改造过的部分，不将尚未完成的上游功能作为个人实现成果。**

接下来所有新增能力都会以独立 Git commit 记录，例如：

```text
feat: add reviewer agent
feat: add review retry loop
feat: add experience memory with StoreBackend
```

---

## Acknowledgements

本项目基于以下开源项目进行学习与二次开发：

- [`didilili/deepsearch-agents`](https://github.com/didilili/deepsearch-agents)
- [`didilili/ai-agents-from-zero`](https://github.com/didilili/ai-agents-from-zero)

感谢原作者提供 DeepAgents 多智能体研究系统的完整工程示例与教程。

后续改造内容与新增实现将在本仓库中持续更新。
