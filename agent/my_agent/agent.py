from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from my_agent.db import db_query_tool, list_tables_tool, get_schema_tool
from my_agent.state import State
from my_agent.nodes import (
    first_tool_call,
    create_tool_node_with_fallback,
    model_check_query,
    query_gen_node,
    final_answer_node,
)
from typing import Literal


def should_continue(
    state: State,
) -> Literal["final_answer", "correct_query", "query_gen"]:
    print("Evaluating should_continue edge condition")
    messages = state["messages"]
    last_message = messages[-1]
    if getattr(last_message, "tool_calls"):
        print("Edge leads to: final_answer")
        return "final_answer"
    if last_message.content.startswith("Error:"):
        print("Edge leads to: query_gen")
        return "query_gen"
    else:
        print("Edge leads to: correct_query")
        return "correct_query"


workflow = StateGraph(State)

workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("correct_query", model_check_query)
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))
workflow.add_node("final_answer", final_answer_node)

workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges("query_gen", should_continue)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")
workflow.add_edge("final_answer", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
