from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from my_agent.query_check import query_check, query_gen
from my_agent.state import State
import json


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    print("Running node: first_tool_call")
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """

    print("Running node: model_check_query")
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}


def query_gen_node(state: State) -> dict[str, list[AIMessage]]:
    print("Running node: query_gen")

    message = query_gen.invoke(state)

    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                        tool_call_id=tc["id"],
                    )
                )

    return {"messages": [message] + tool_messages}


def final_answer_node(state: State) -> dict[str, list[AIMessage]]:
    print("Running node: final_answer")

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_call"):
        print("Error: No valid final answer found.")
        return {"messages": [AIMessage(content="Error: No valid final answer found.")]}

    arguments = json.loads(
        last_message.additional_kwargs["tool_calls"][0]["function"]["arguments"]
    )
    final_answer_content = arguments.get("final_answer", "No final answer found.")

    print(f"Final answer: {final_answer_content}")
    return {"messages": [AIMessage(content=final_answer_content)]}
