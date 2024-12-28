from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_groq import ChatGroq

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
    """
    print(f"Executing query: {query}")
    result = db.run_no_throw(query)

    if not result:
        print("Error: Query failed. Please rewrite your query and try again.")
        return "Error: Query failed. Please rewrite your query and try again."

    print(f"Query result: {result}")
    return result
