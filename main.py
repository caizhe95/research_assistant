from datetime import datetime

from config import DEFAULT_USER_ID, RESEARCH_TOPIC
from graph_builder import create_research_assistant
from state import build_turn_input

def run_research_turn(assistant, question: str, thread_id: str, user_id: str) -> dict:
    print("\n" + "=" * 60)
    print("启动研究任务")
    print("=" * 60)
    print(f"用户问题: {question}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"线程 ID : {thread_id}")
    print(f"用户 ID : {user_id}")

    result = assistant.invoke(
        build_turn_input(question, user_id),
        config={"configurable": {"thread_id": thread_id}},
    )

    print("\n" + "=" * 60)
    print("最终回答")
    print("=" * 60)
    print(result.get("final_answer", "回答生成失败"))

    citations = result.get("citations", [])
    if citations:
        print("\n" + "-" * 60)
        print("引用来源")
        print("-" * 60)
        for citation in citations:
            print(
                f"{citation['id']} {citation['title']} | {citation['source']} | {citation.get('url') or '无链接'}"
            )

    print("\n" + "-" * 60)
    print("本轮信息")
    print("-" * 60)
    print(f"  - 当前阶段: {result.get('current_phase', 'unknown')}")
    print(f"  - 改写次数: {result.get('iteration_count', 0)}")
    print(f"  - 记忆命中数: {len(result.get('memory_hits', []))}")
    print(f"  - 证据条数: {len(result.get('citations', []))}")
    print(f"  - 回答字数: {len(result.get('final_answer', ''))}")

    react_trace = result.get("react_trace", [])
    if react_trace:
        print("\n" + "-" * 60)
        print("ReAct 轨迹")
        print("-" * 60)
        for step in react_trace:
            print(f"  - [{step.get('phase', 'unknown')}] {step.get('note', '')}")

    return result

def demonstrate_project():
    print("=" * 60)
    print("可切换主题的研究助手项目")
    print("=" * 60)
    print(f"当前研究主题: {RESEARCH_TOPIC}")

    user_id = DEFAULT_USER_ID
    thread_id = "research_demo_thread"

    assistant = create_research_assistant(user_id=user_id)

    demo_questions = [
        f"我的研究主题是“{RESEARCH_TOPIC}”。请先给我一个研究范围定义和 3 个关键子问题。",
        f"围绕“{RESEARCH_TOPIC}”，请结合知识库内容给出一个可执行的调研计划。",
        f"针对“{RESEARCH_TOPIC}”，请总结目前证据中的主要结论、争议点和下一步数据需求。",
    ]

    for index, question in enumerate(demo_questions, 1):
        print(f"\n{'#' * 60}")
        print(f"# 演示问题 {index}")
        print(f"{'#' * 60}")
        run_research_turn(assistant, question, thread_id, user_id)

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)

def main():
    demonstrate_project()

    print("\n你也可以这样使用：")
    print("-" * 60)
    print(
        """
from graph_builder import create_research_assistant
from state import build_turn_input

assistant = create_research_assistant(user_id="my_user")
config = {"configurable": {"thread_id": "session_001"}}

result = assistant.invoke(
    {
        **build_turn_input("请帮我调研我的主题并输出带引用结论", "my_user"),
        "include_react_trace_in_answer": True,
    },
    config=config,
)

print(result["final_answer"])
"""
    )


if __name__ == "__main__":
    main()
