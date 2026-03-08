from typing import Literal

from pydantic import BaseModel, Field

class ResearchPlan(BaseModel):
    intent: str = Field(description="用户真正想解决的问题")
    need_retrieval: bool = Field(description="是否需要调用工具检索资料")
    sub_questions: list[str] = Field(default_factory=list, description="拆解后的研究子问题")
    preferred_tools: list[str] = Field(default_factory=list, description="建议优先使用的工具名称")
    answer_style: str = Field(description="建议采用的回答风格")
    risk_reminder: str = Field(description="本次回答最需要注意的风险点，如资料不足、时效性等")


class RelevanceGrade(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="相关则为 yes，否则为 no")
    reason: str = Field(description="判断理由，要求简洁")


class MemoryDecision(BaseModel):
    should_store: bool = Field(description="是否值得写入长期记忆")
    memory_type: Literal["preference", "goal", "fact", "style"] = Field(
        description="记忆类型：偏好、目标、事实、风格"
    )
    content: str = Field(description="要保存的记忆内容")
    tags: list[str] = Field(default_factory=list, description="记忆标签")
