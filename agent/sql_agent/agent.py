from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import Literal, Annotated, List
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()


db = SQLDatabase.from_uri(r"sqlite:///data/data.db")


def db_query_tool(query: str) -> str:
    """
    Executes a SQL query against the database and returns the result.

    Args:
        query (str): The SQL query to execute.

    Returns:
        str: Query result or an error message if the query fails.
    """
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result


llm = ChatGroq(model="llama-3.3-70b-versatile")


def get_context() -> str:
    """
    Retrieves information about the database schema to provide context for query generation.

    Returns:
        str: Database schema information.
    """
    return db.get_table_info()


class AgentState(TypedDict):
    """
    Defines the structure of the agent's state.

    Attributes:
        messages (List[BaseMessage]): List of messages exchanged during the interaction.
        user_query (str): The user-provided query.
        sql_query (str): The generated SQL query.
        query_results (str): Results from executing the SQL query.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    sql_query: str
    query_results: str


def check_relevance(state: AgentState) -> Literal["YES", "NO"]:
    """
    Determines whether the user's query is relevant to the database context.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        Literal["YES", "NO"]: 'YES' if the query is relevant, otherwise 'NO'.
    """

    class QueryRelevance(BaseModel):
        query_relevant: str = Field(
            description="Relevance of user query to database context. The response must be 'YES' or 'NO'."
        )

    relevance_prompt = PromptTemplate.from_template(
        """
        You are a relevance evaluation assistant. Your task is to determine if a user's query is relevant to a given database context. 
        Respond with 'YES' if the query is relevant, and 'NO' if it is not.

        Database Context: {context}.

        Example Queries (may not be related to context):
        1. "What were the total sales in 2010?" -> YES
        2. "Who is the CEO of the company?" -> NO
        3. "Which sales agent had the highest sales in 2009?" -> YES
        4. "What is the weather today?" -> NO

        Now evaluate the following query and respond with 'YES' or 'NO':
        {query}
        """
    )

    model = llm.with_structured_output(QueryRelevance)
    relevance_generator = relevance_prompt | model

    inputs = {"context": get_context(), "query": state["messages"][-1].content}
    relevance = relevance_generator.invoke(inputs)

    return relevance.query_relevant


def sql_query_generation(state: AgentState) -> AgentState:
    """
    Generates a SQL query based on the user's query and database context.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: Updated state with the generated SQL query.
    """
    sql_generation_prompt = PromptTemplate.from_template(
        """
        You are a highly skilled SQL query generator. Your task is to generate a valid SQL query based on a given database context and user query. 
        The SQL query should adhere to standard SQL syntax and align with the details provided in the database context.

        Database Context: {context}

        Now, evaluate the following input and generate an SQL query:

        User Query: {query}

        Output:
        Provide only the SQL query as a response, without any additional formatting or explanation.
        """
    )

    state["user_query"] = state["messages"][-1].content

    inputs = {
        "context": get_context(),
        "query": state["user_query"],
    }

    sql_query_generator = sql_generation_prompt | llm | StrOutputParser()
    state["sql_query"] = sql_query_generator.invoke(inputs)

    print(state["messages"])

    return state


def sql_query_execution(state: AgentState) -> AgentState:
    """
    Executes the SQL query and retrieves the results.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: Updated state with the SQL query results.
    """
    state["query_results"] = db_query_tool(state["sql_query"])
    return state


def answer_generation(state: AgentState) -> AgentState:
    """
    Generates a natural language response based on the user query and SQL query results.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: Updated state with the generated response.
    """
    answer_generation_prompt = PromptTemplate.from_template(
        """
        You are a highly skilled assistant responsible for generating user-friendly answers based on a user query and the SQL query's output.
        You will use the provided SQL output to craft a clear, concise, and helpful response that directly answers the user's query.

        User Query: {query}

        SQL Query Output:
        {sql_output}

        Instructions:
        - Analyze the SQL query output.
        - Generate a response that answers the user query in natural language.
        - Be specific and provide numerical values or details where applicable.

        Output:
        Provide only the answer to the user query in plain text, without any additional formatting or explanation.
        """
    )

    inputs = {
        "query": state["user_query"],
        "sql_output": state["query_results"],
    }

    answer_generator = answer_generation_prompt | llm
    return {**state, "messages": state["messages"] + [answer_generator.invoke(inputs)]}


def query_not_relevant(state: AgentState) -> AgentState:
    """
    Handles cases where the user's query is not relevant to the database context.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: Updated state with a message indicating irrelevance.
    """
    return {
        **state,
        "messages": state["messages"]
        + [AIMessage(content="Given user query is not relevant to database context.")],
    }


workflow = StateGraph(AgentState)

workflow.add_node("sql_query_generation", sql_query_generation)
workflow.add_node("sql_query_execution", sql_query_execution)
workflow.add_node("answer_generation", answer_generation)
workflow.add_node("query_not_relevant", query_not_relevant)

workflow.add_conditional_edges(
    START, check_relevance, {"YES": "sql_query_generation", "NO": "query_not_relevant"}
)
workflow.add_edge("query_not_relevant", END)
workflow.add_edge("sql_query_generation", "sql_query_execution")
workflow.add_edge("sql_query_execution", "answer_generation")
workflow.add_edge("answer_generation", END)

graph = workflow.compile(checkpointer=MemorySaver())
