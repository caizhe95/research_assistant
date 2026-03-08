from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from config import (
    DEFAULT_USER_ID,
    KNOWLEDGE_DIR,
    MAX_REACT_STEPS,
    MAX_REWRITE_TIMES,
    MEMORY_DIR,
    MEMORY_TOP_K,
)
from context_manager import ConversationContextManager
from knowledge_base import KnowledgeBase
from llm import get_chat_model, get_embeddings
from memory_store import ResearchMemoryStore
from models import MemoryDecision, RelevanceGrade, ResearchPlan
from prompts import (
    ANSWER_PROMPT,
    GRADE_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    PLANNING_PROMPT,
    RESEARCH_ASSISTANT_SYSTEM_PROMPT,
    REWRITE_PROMPT,
)
from state import ResearchAssistantState
from tools import create_tools
from utils import (
    build_evidence_block,
    format_memory_block,
    format_plan_block,
    get_last_user_message,
    parse_tool_payloads,
)

def create_research_assistant(user_id: str = DEFAULT_USER_ID):
    planner_model = get_chat_model(temperature=0.0)
    reasoner_model = get_chat_model(temperature=0.1)
    writer_model = get_chat_model(temperature=0.3)
    summarizer_model = get_chat_model(temperature=0.0)

    embeddings = get_embeddings()
    knowledge_base = KnowledgeBase(KNOWLEDGE_DIR, embeddings)
    knowledge_base.build()

    memory_store = ResearchMemoryStore(MEMORY_DIR, embeddings)
    context_manager = ConversationContextManager(summarizer_model)
    tools = create_tools(knowledge_base, memory_store, user_id)

    def append_react_trace(state: ResearchAssistantState, phase: str, note: str) -> list[dict]:
        trace = list(state.get("react_trace", []))
        trace.append({"phase": phase, "note": note})
        return trace

    def render_react_trace(trace: list[dict]) -> str:
        if not trace:
            return "无"
        lines = []
        for index, step in enumerate(trace, 1):
            lines.append(f"{index}. [{step.get('phase', 'unknown')}] {step.get('note', '')}")
        return "\n".join(lines)

    def prepare_context_node(state: ResearchAssistantState) -> dict:
        print("\n" + "=" * 50)
        print("上下文准备阶段")

        current_query = get_last_user_message(state["messages"])
        conversation_summary = context_manager.summarize_history(state["messages"])
        memory_hits = memory_store.search(
            user_id=user_id,
            query=current_query,
            limit=MEMORY_TOP_K,
        )

        print(f"   当前问题: {current_query}")
        print(f"   命中长期记忆: {len(memory_hits)} 条")

        return {
            "current_query": current_query,
            "rewritten_query": "",
            "active_question": current_query,
            "conversation_summary": conversation_summary,
            "memory_hits": memory_hits,
            "retrieved_chunks": [],
            "citations": [],
            "final_answer": "",
            "current_phase": "planning",
            "iteration_count": 0,
            "react_step": 0,
            "react_trace": append_react_trace(
                state,
                phase="thought",
                note=f"识别用户问题并整理上下文：{current_query}",
            ),
        }

    def planning_node(state: ResearchAssistantState) -> dict:
        print("\n" + "-" * 50)
        print("研究规划阶段")

        memory_block = format_memory_block(state.get("memory_hits", []))
        prompt = PLANNING_PROMPT.format(
            question=state["active_question"],
            conversation_summary=state.get("conversation_summary", ""),
            memory_block=memory_block,
        )

        plan = planner_model.with_structured_output(ResearchPlan).invoke(
            [HumanMessage(content=prompt)]
        )
        plan_data = plan.model_dump()

        print(f"   用户意图: {plan_data.get('intent', 'N/A')}")
        print(f"   是否建议检索: {plan_data.get('need_retrieval', False)}")
        print(f"   子问题数量: {len(plan_data.get('sub_questions', []))}")

        return {
            "research_plan": plan_data,
            "current_phase": "reasoning",
            "react_trace": append_react_trace(
                state,
                phase="thought",
                note=f"形成研究计划，need_retrieval={plan_data.get('need_retrieval', False)}",
            ),
        }

    def generate_query_or_respond_node(state: ResearchAssistantState) -> dict:
        print("\n" + "-" * 50)
        print("推理决策阶段")

        summary = state.get("conversation_summary", "")
        memory_block = format_memory_block(state.get("memory_hits", []))
        plan_block = format_plan_block(state.get("research_plan", {}))

        system_prompt = RESEARCH_ASSISTANT_SYSTEM_PROMPT.format(
            conversation_summary=summary,
            memory_block=memory_block,
            plan_block=plan_block,
        )

        reasoning_messages = context_manager.build_reasoning_messages(
            messages=state["messages"],
            question=state["active_question"],
        )

        response = reasoner_model.bind_tools(tools).invoke(
            [SystemMessage(content=system_prompt)] + reasoning_messages
        )

        has_tool_calls = bool(getattr(response, "tool_calls", []))
        print(f"   工具调用: {'是' if has_tool_calls else '否'}")
        print(f"   ReAct 步数: {state.get('react_step', 0) + 1}/{MAX_REACT_STEPS}")

        if has_tool_calls:
            tool_names = [call.get("name", "unknown_tool") for call in getattr(response, "tool_calls", [])]
            react_trace = append_react_trace(
                state,
                phase="action",
                note=f"决定调用工具：{', '.join(tool_names)}",
            )
        else:
            react_trace = append_react_trace(
                state,
                phase="thought",
                note="判断可直接回答，无需外部工具。",
            )

        return {
            "messages": [response],
            "current_phase": "tool_or_answer",
            "react_step": state.get("react_step", 0) + 1,
            "react_trace": react_trace,
        }

    def route_after_reasoning(state: ResearchAssistantState) -> Literal["tools", "direct_answer", "max_step_answer"]:
        last_message = state["messages"][-1]
        if state.get("react_step", 0) >= MAX_REACT_STEPS:
            print(f"   达到最大 ReAct 步数 {MAX_REACT_STEPS}，停止继续调用工具")
            return "max_step_answer"
        if getattr(last_message, "tool_calls", []):
            return "tools"
        return "direct_answer"

    def finalize_direct_answer_node(state: ResearchAssistantState) -> dict:
        print("\n" + "-" * 50)
        print("直接回答收尾阶段")

        last_message = state["messages"][-1]
        answer = last_message.content if isinstance(last_message, AIMessage) else ""
        trace = append_react_trace(
            state,
            phase="finish",
            note="直接回答路径完成。",
        )
        if state.get("include_react_trace_in_answer", True):
            answer = f"{answer}\n\n---\nReAct 轨迹\n{render_react_trace(trace)}"

        return {
            "final_answer": answer,
            "citations": [],
            "retrieved_chunks": [],
            "current_phase": "memory_update",
            "react_trace": trace,
        }

    def grade_documents(state: ResearchAssistantState) -> Literal["generate_answer", "rewrite_question"]:
        print("\n" + "-" * 50)
        print("检索结果评估阶段")

        tool_payloads = parse_tool_payloads(state["messages"])
        evidence_block, _, _ = build_evidence_block(tool_payloads)
        evidence_count = sum(len(payload.get("items", [])) for payload in tool_payloads)
        state["react_trace"] = append_react_trace(
            state,
            phase="observation",
            note=f"工具返回证据 {evidence_count} 条。",
        )

        # 没拿到有效检索结果时，先尝试改写一次查询
        if not tool_payloads or evidence_block == "暂无可用外部证据。":
            print("   未拿到有效结果，准备改写问题")
            state["react_trace"] = append_react_trace(
                state,
                phase="observation",
                note="检索结果不足，进入问题改写。",
            )
            if state.get("iteration_count", 0) >= MAX_REWRITE_TIMES:
                return "generate_answer"
            return "rewrite_question"

        question = state.get("active_question", state.get("current_query", ""))
        prompt = GRADE_PROMPT.format(
            question=question,
            context=evidence_block[:3000],
        )

        result = planner_model.with_structured_output(RelevanceGrade).invoke(
            [HumanMessage(content=prompt)]
        )

        print(f"   相关性判断: {result.binary_score}")
        print(f"   判断原因: {result.reason}")
        state["react_trace"] = append_react_trace(
            state,
            phase="observation",
            note=f"相关性评估={result.binary_score}，原因：{result.reason}",
        )

        if result.binary_score == "yes" or state.get("iteration_count", 0) >= MAX_REWRITE_TIMES:
            return "generate_answer"
        return "rewrite_question"

    def rewrite_question_node(state: ResearchAssistantState) -> dict:
        print("\n" + "-" * 50)
        print("检索问题改写阶段")

        plan_block = format_plan_block(state.get("research_plan", {}))
        prompt = REWRITE_PROMPT.format(
            question=state.get("active_question", state.get("current_query", "")),
            plan_block=plan_block,
        )
        response = planner_model.invoke([HumanMessage(content=prompt)])
        rewritten_query = response.content.strip()

        print(f"   改写后问题: {rewritten_query}")

        return {
            "rewritten_query": rewritten_query,
            "active_question": rewritten_query,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "current_phase": "reasoning",
            "react_trace": append_react_trace(
                state,
                phase="thought",
                note=f"证据不足，改写检索问题：{rewritten_query}",
            ),
        }

    def generate_answer_node(state: ResearchAssistantState) -> dict:
        print("\n" + "-" * 50)
        print("最终答案生成阶段")

        tool_payloads = parse_tool_payloads(state["messages"])
        evidence_block, citations, retrieved_chunks = build_evidence_block(tool_payloads)

        summary = state.get("conversation_summary", "")
        memory_block = format_memory_block(state.get("memory_hits", []))
        plan_block = format_plan_block(state.get("research_plan", {}))
        question = state.get("active_question", state.get("current_query", ""))

        prompt = ANSWER_PROMPT.format(
            question=question,
            conversation_summary=summary,
            memory_block=memory_block,
            plan_block=plan_block,
            evidence_block=evidence_block,
        )

        answer = writer_model.invoke([HumanMessage(content=prompt)])

        print(f"   证据数量: {len(citations)}")
        print(f"   回答字数: {len(answer.content)}")

        return {
            "messages": [answer],
            "final_answer": (
                f"{answer.content}\n\n---\nReAct 轨迹\n"
                f"{render_react_trace(append_react_trace(state, phase='finish', note=f'基于 {len(citations)} 条证据生成最终回答。'))}"
                if state.get("include_react_trace_in_answer", True)
                else answer.content
            ),
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "current_phase": "memory_update",
            "react_trace": append_react_trace(
                state,
                phase="finish",
                note=f"基于 {len(citations)} 条证据生成最终回答。",
            ),
        }

    def memory_update_node(state: ResearchAssistantState) -> dict:
        print("\n" + "-" * 50)
        print("长期记忆更新阶段")

        question = state.get("current_query", "")
        answer = state.get("final_answer", "")

        prompt = MEMORY_EXTRACTION_PROMPT.format(
            question=question,
            answer=answer,
        )

        decision = planner_model.with_structured_output(MemoryDecision).invoke(
            [HumanMessage(content=prompt)]
        )

        if decision.should_store and decision.content.strip():
            memory_item = memory_store.add_memory(
                user_id=user_id,
                content=decision.content,
                tags=decision.tags,
                memory_type=decision.memory_type,
            )
            print(f"   已写入长期记忆: {memory_item['content']}")
        else:
            print("   本轮无新增长期记忆")

        return {
            "current_phase": "completed",
        }

    workflow = StateGraph(ResearchAssistantState)

    workflow.add_node("prepare_context", prepare_context_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond_node)
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("finalize_direct_answer", finalize_direct_answer_node)
    workflow.add_node("rewrite_question", rewrite_question_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("memory_update", memory_update_node)

    workflow.add_edge(START, "prepare_context")
    workflow.add_edge("prepare_context", "planning")
    workflow.add_edge("planning", "generate_query_or_respond")

    workflow.add_conditional_edges(
        "generate_query_or_respond",
        route_after_reasoning,
        {
            "tools": "retrieve",
            "direct_answer": "finalize_direct_answer",
            "max_step_answer": "generate_answer",
        },
    )

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        },
    )

    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    workflow.add_edge("finalize_direct_answer", "memory_update")
    workflow.add_edge("generate_answer", "memory_update")
    workflow.add_edge("memory_update", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    return graph
