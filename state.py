from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages


class ResearchAssistantState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    current_query: str
    rewritten_query: str
    active_question: str
    conversation_summary: str
    memory_hits: list[dict]
    research_plan: dict
    retrieved_chunks: list[dict]
    citations: list[dict]
    final_answer: str
    current_phase: str
    iteration_count: int
    react_step: int
    include_react_trace_in_answer: bool
    react_trace: list[dict]


def build_turn_input(question: str, user_id: str) -> ResearchAssistantState:
    return {
        "messages": [HumanMessage(content=question)],
        "user_id": user_id,
        "current_query": question,
        "rewritten_query": "",
        "active_question": question,
        "conversation_summary": "",
        "memory_hits": [],
        "research_plan": {},
        "retrieved_chunks": [],
        "citations": [],
        "final_answer": "",
        "current_phase": "prepare_context",
        "iteration_count": 0,
        "react_step": 0,
        "include_react_trace_in_answer": True,
        "react_trace": [],
    }
