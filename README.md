# Transformer 方向研究助手项目

这是一个在原始单文件教学项目基础上重构的 **个人项目版研究助手**。  
当前默认研究主题为 **Transformer 架构与大模型应用**，目标是做成一个 **结构清晰、能跑起来、能快速切换主题与知识源** 的项目。

---

## 1. 项目定位

这个项目重点展示以下能力：

- 基于 **LangGraph** 设计有状态工作流
- 使用 **LangChain** 管理模型、工具、结构化输出
- 使用 **DeepSeek** 作为对话模型
- 使用 **BGE Small** 作为 embedding 模型
- 实现 **短期记忆 + 长期记忆**
- 实现 **上下文管理**
- 实现 **工具调用**
- 实现 **外挂知识源**
- 实现 **按需 ReAct 的 Agentic RAG**
- 输出时带 **引用来源**

这个复杂度很适合个人项目：
- 有明确架构
- 有可讲的工程细节
- 但不会复杂到像企业内部系统那样难以解释

---

## 2. 技术栈

- `langchain>=1.0.0`
- `langgraph>=1.0.0`
- `langchain-deepseek`
- `langchain-huggingface`
- `langchain-text-splitters`
- `sentence-transformers`
- `arxiv`
- `python-dotenv`

---

## 3. 功能概览

### 3.1 短期记忆
使用 LangGraph 的 `InMemorySaver` 保存线程级状态。  
同一个 `thread_id` 下，助手可以记住前面对话内容。

### 3.2 长期记忆
项目额外实现了一个 `JSON + embedding` 的轻量级长期记忆：
- 把用户稳定偏好、长期目标保存到 `data/memory/`
- 下次对话前先检索相关记忆
- 让回答更个性化

### 3.3 上下文管理
引入 `context_manager.py`：
- 压缩较长历史对话
- 只保留最近关键轮次
- 避免上下文越来越长导致信息污染

### 3.4 工具定义
项目定义了 3 个工具：

- `search_local_knowledge`：检索本地知识库
- `search_arxiv_papers`：搜索 arXiv 论文摘要
- `recall_user_memory`：检索用户长期记忆

### 3.5 外挂知识源（当前为 Transformer 主题）
外挂知识源分两类：
1. `data/knowledge/` 下的本地知识文件
2. arXiv 的公开论文摘要

默认本地知识文件已切换为 Transformer 方向（总览、核心事实、风险争议、研究问题）。

### 3.6 Agentic RAG
流程不是“无脑检索后回答”，而是：

1. 先规划问题
2. 再让模型决定是否需要工具
3. 调用检索工具
4. 对检索结果做相关性判断
5. 如果不够相关，就自动改写问题后重试一次
6. 最后再生成带引用的回答

这套流程在代码中映射为 **完整 ReAct 回路（Thought / Action / Observation）**：
- `Thought`：`prepare_context`、`planning`、`generate_query_or_respond`
- `Action`：`ToolNode(tools)` 执行检索工具
- `Observation`：`grade_documents` 对工具返回结果做证据数量与相关性判断
- `Thought(修正)`：若证据不足，`rewrite_question` 改写问题后回到 `generate_query_or_respond`
- `Finish`：`generate_answer` 或 `finalize_direct_answer`

并且支持：
- `MAX_REACT_STEPS`：限制 ReAct 循环最大步数，避免无限迭代
- `react_trace`：记录每一步轨迹（thought/action/observation/finish）
- `include_react_trace_in_answer`：是否把轨迹附加到最终回答（默认开启，方便演示）

说明：当前策略是“按需 ReAct”，即模型判断不需要工具时可直接回答；需要时才进入 ReAct 循环。

---

## 4. 项目结构

```text
research_assistant_llm_project/
├── main.py
├── config.py
├── llm.py
├── models.py
├── state.py
├── prompts.py
├── utils.py
├── knowledge_base.py
├── memory_store.py
├── context_manager.py
├── tools.py
├── graph_builder.py
├── requirements.txt
├── .env.example
└── data
    ├── knowledge
    │   ├── 01_topic_overview.md
    │   ├── 02_key_facts.md
    │   ├── 03_risks_and_debates.md
    │   └── 04_research_questions.md
    └── memory
```

---

## 5. 运行方式

### 5.1 安装依赖
```bash
pip install -r requirements.txt
```

### 5.2 配置环境变量
复制 `.env.example` 为 `.env`，然后填入：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_CHAT_MODEL=deepseek-chat
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
USER_ID=demo_user
RESEARCH_TOPIC=Transformer架构与大模型应用
KNOWLEDGE_DIR=./data/knowledge
```

- `RESEARCH_TOPIC`：演示问题中的默认研究主题（建议与你的知识文件主题一致）
- `KNOWLEDGE_DIR`：知识库目录，可指向任意本地文档目录
- `MAX_REACT_STEPS`：在 `config.py` 中配置，当前为 `2`

### 5.3 运行项目
```bash
python main.py
```

---

## 6. 可继续扩展的方向

如果你还想继续增强，可以加：

- Chroma / FAISS 持久化向量库
- 研究报告导出 Markdown / PDF
- LangSmith tracing
- 更细的评估脚本
- Web 搜索工具（如 Tavily / DuckDuckGo）

---

## 7. 工程边界

这个项目故意不做以下内容：

- 多智能体协作
- 复杂权限系统
- 企业级数据库部署
- 大规模知识库增量更新
- 在线服务化接口

原因很简单：  
**项目实践最重要的是“目标清晰 + 能跑 + 有取舍”，不是盲目堆复杂度。**

---

## 8. 当前默认配置（2026-03）

为方便直接演示，仓库当前默认配置如下：

1. 研究主题：`Transformer架构与大模型应用`
2. 本地知识源：`data/knowledge/` 下 4 份 Transformer 主题文档
3. ReAct 策略：按需触发（不是强制每轮调用工具）
4. ReAct 循环上限：`MAX_REACT_STEPS = 2`
5. 输出内容：默认附带 `ReAct 轨迹`（可通过状态字段关闭）
