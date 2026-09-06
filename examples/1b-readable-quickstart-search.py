"""
DeepAgents quickstart with readable console output.

This is a learning-friendly version of 1-deep-agent-quickstart-search.py.
It keeps the same Agent flow, but prints only:
1. user question
2. model tool decision
3. tool result summary
4. final answer
"""

import os
from typing import Literal

from deepagents import create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tavily import TavilyClient

load_dotenv(find_dotenv())

llm_name = os.getenv("LLM_QWEN_MAX")
tavily_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_key)


@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["news", "finance", "general"] = "general",
    include_raw_content: bool = False,
):
    """Search public web information with Tavily."""
    print(f"\n[Tool Start] internet_search")
    print(f"query={query}, max_results={max_results}, topic={topic}")
    return tavily_client.search(
        query=query,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )


llm = init_chat_model(model=llm_name, model_provider="openai")

deep_agent = create_deep_agent(
    model=llm,
    tools=[internet_search],
    subagents=[],
    system_prompt="""
    你是一名严谨的研究员，可以使用 internet_search 工具检索网络信息。
    请根据检索结果进行归纳、分析和交叉验证，生成一份结构清晰、信息可靠的中文报告。
    """,
)

question = "请查询地震预警和地震反投影领域的热门论文，并整理为一份简要报告。"
result = deep_agent.invoke({"messages": [{"role": "user", "content": question}]})

print("\n================ DeepAgent Execution ================")

for index, message in enumerate(result["messages"], start=1):
    print(f"\n[{index}] {message.__class__.__name__}")

    if isinstance(message, HumanMessage):
        print(f"User: {message.content}")

    elif isinstance(message, AIMessage) and message.tool_calls:
        print("Model decided to call tool(s):")
        for tool_call in message.tool_calls:
            print(f"- {tool_call['name']}: {tool_call['args']}")

    elif isinstance(message, ToolMessage):
        content = str(message.content)
        preview = content[:500].replace("\n", " ")
        print(f"Tool returned: {message.name}")
        print(f"Preview: {preview}...")

    elif isinstance(message, AIMessage) and message.content:
        print("Final answer:")
        print(message.content)

print("\n================ Final Answer Only ================")
print(result["messages"][-1].content)
