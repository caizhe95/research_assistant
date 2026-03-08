from langchain_core.messages import HumanMessage

from config import MAX_HISTORY_TURNS
from prompts import CONTEXT_SUMMARY_PROMPT

class ConversationContextManager:
    def __init__(self, summarizer_model):
        self.summarizer_model = summarizer_model

    def summarize_history(self, messages: list, keep_last: int = MAX_HISTORY_TURNS) -> str:
        # 只保留人类和助手消息，跳过工具调用消息
        dialogue_messages = [
            message
            for message in messages
            if getattr(message, "type", "") in {"human", "ai"}
            and not getattr(message, "tool_calls", [])
        ]

        if len(dialogue_messages) <= keep_last:
            return ""

        history_text = []
        for message in dialogue_messages[:-keep_last]:
            role = "用户" if getattr(message, "type", "") == "human" else "助手"
            history_text.append(f"{role}：{message.content}")

        prompt = CONTEXT_SUMMARY_PROMPT.format(history="\n".join(history_text))
        response = self.summarizer_model.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def build_reasoning_messages(self, messages: list, question: str) -> list:
        dialogue_messages = [
            message
            for message in messages
            if getattr(message, "type", "") in {"human", "ai"}
            and not getattr(message, "tool_calls", [])
        ]
        dialogue_messages = dialogue_messages[-MAX_HISTORY_TURNS:]

        if dialogue_messages and getattr(dialogue_messages[-1], "type", "") == "human":
            dialogue_messages = dialogue_messages[:-1] + [HumanMessage(content=question)]
        else:
            dialogue_messages = dialogue_messages + [HumanMessage(content=question)]

        return dialogue_messages
