from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import Any, Annotated, Literal
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
import json


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


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


db = SQLDatabase.from_uri("sqlite:///data/data.db")
toolkit = SQLDatabaseToolkit(db=db, llm=ChatGroq(model="llama-3.3-70b-versatile"))
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")


@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    print(f"Executing query: {query}")
    result = db.run_no_throw(query)

    if not result:
        print("Error: Query failed. Please rewrite your query and try again.")
        return "Error: Query failed. Please rewrite your query and try again."

    print(f"Query result: {result}")
    return result


query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)
query_check = query_check_prompt | ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0
).bind_tools([db_query_tool], tool_choice="required")


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


class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""

    final_answer: str = Field(..., description="The final answer to the user")


query_gen_system = """You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

When generating the query:

Output the SQL query that answers the input question without a tool call.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

If you get an error while executing a query, rewrite the query and try again.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0
).bind_tools([SubmitFinalAnswer])


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


model_get_schema = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(
    [get_schema_tool]
)

workflow = StateGraph(State)

workflow.add_node("first_tool_call", first_tool_call)
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("correct_query", model_check_query)
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))
workflow.add_node("final_answer", final_answer_node)

workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges(
    "query_gen",
    should_continue,
)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")
workflow.set_finish_point("final_answer")

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)
