import json
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def safe_parse_json(text: str, default: Any = None) -> Any:
    if default is None:
        default = {}

    if not text:
        return default

    content = text.strip()

    if "```json" in content:
        try:
            content = content.split("```json", 1)[1].split("```", 1)[0]
        except IndexError:
            return default
    elif "```" in content:
        try:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        except IndexError:
            return default

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return default


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_last_user_message(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
        if getattr(message, "type", "") == "human":
            return getattr(message, "content", "")
    return ""


def get_latest_tool_messages(messages: list) -> list[ToolMessage]:
    tool_messages = []

    for message in reversed(messages):
        message_type = getattr(message, "type", "")
        if message_type == "tool" or isinstance(message, ToolMessage):
            tool_messages.append(message)
            continue

        if tool_messages and (message_type == "ai" or isinstance(message, AIMessage)):
            break

    return list(reversed(tool_messages))


def parse_tool_payloads(messages: list) -> list[dict]:
    payloads = []

    for message in get_latest_tool_messages(messages):
        payload = safe_parse_json(getattr(message, "content", ""), default={})
        if payload:
            payloads.append(payload)

    return payloads


def build_evidence_block(tool_payloads: list[dict]) -> tuple[str, list[dict], list[dict]]:
    evidence_lines = []
    citations = []
    retrieved_chunks = []
    index = 1

    for payload in tool_payloads:
        items = payload.get("items", [])

        for item in items:
            chunk = {
                "id": f"chunk_{index}",
                "title": item.get("title", f"来源{index}"),
                "source": item.get("source", "unknown"),
                "url": item.get("url"),
                "content": item.get("content") or item.get("snippet") or item.get("summary") or "",
                "source_type": item.get("source_type", payload.get("tool", "tool")),
            }
            retrieved_chunks.append(chunk)

            citations.append(
                {
                    "id": f"[{index}]",
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "url": chunk["url"],
                }
            )

            # 统一成可读文本，后面给模型做回答用
            evidence_lines.append(
                "\n".join(
                    [
                        f"[{index}] 标题：{chunk['title']}",
                        f"来源：{chunk['source']}",
                        f"链接：{chunk['url'] or '无'}",
                        f"内容：{chunk['content'][:500]}",
                    ]
                )
            )
            index += 1

    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "暂无可用外部证据。"
    return evidence_block, citations, retrieved_chunks


def format_memory_block(memory_hits: list[dict]) -> str:
    if not memory_hits:
        return "暂无长期记忆。"

    lines = []
    for item in memory_hits:
        tags = ", ".join(item.get("tags", []))
        lines.append(
            f"- 类型：{item.get('memory_type', 'unknown')}；内容：{item.get('content', '')}；标签：{tags or '无'}"
        )

    return "\n".join(lines)


def format_plan_block(plan: dict) -> str:
    if not plan:
        return "暂无研究计划。"

    lines = [
        f"用户意图：{plan.get('intent', '')}",
        f"是否建议检索：{plan.get('need_retrieval', False)}",
        f"回答风格：{plan.get('answer_style', '')}",
        f"风险提醒：{plan.get('risk_reminder', '')}",
        "子问题：",
    ]
    for question in plan.get("sub_questions", []):
        lines.append(f"- {question}")

    if plan.get("preferred_tools"):
        lines.append("建议工具：")
        for tool_name in plan.get("preferred_tools", []):
            lines.append(f"- {tool_name}")

    return "\n".join(lines)
