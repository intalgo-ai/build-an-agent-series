import os
import sys
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_community.tools import QuerySQLDataBaseTool
from typing import Any, Dict, Optional, List
from sqlalchemy import create_engine, text
import logging
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import AgentExecutor, create_openai_functions_agent

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

# Check if all required environment variables are present
if not all([openai_api_key, db_user, db_password, db_host, db_port, db_name]):
    raise ValueError("Missing required environment variables")

# Construct database URI
db_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

class QuerySQLDatabaseTool(QuerySQLDataBaseTool):
    def _run(self, query: str, run_manager: Optional[Any] = None) -> str:
        logger.info(f"Executing query: {query}")
        try:
            result = self.db.run(query)
            logger.info("Query executed successfully")
            return str(result)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return f"Error executing query: {str(e)}"

class CustomSQLDatabase(SQLDatabase):
    def __init__(self, engine, schema=None):
        super().__init__(engine, schema)
        self.engine = engine
        self.schema = schema

    def run(self, command: str, fetch: str = "all") -> Dict[str, Any]:
        logger.info(f"Executing SQL command: {command}")
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(command))
                if fetch == "all":
                    rows = result.fetchall()
                    return {"result": [dict(zip(result.keys(), row)) for row in rows]}
                else:
                    row = result.fetchone()
                    return {"result": dict(zip(result.keys(), row)) if row else None}
        except Exception as e:
            logger.error(f"Error executing SQL command: {str(e)}")
            raise

# Create database engine and test connection
engine = create_engine(db_uri)
def test_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}", exc_info=True)
        raise

test_connection()

# Initialize database, model, and toolkit
db = CustomSQLDatabase(engine, schema=db_name)
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

class CustomSQLDatabaseToolkit(SQLDatabaseToolkit):
    def get_tools(self) -> List[QuerySQLDatabaseTool]:
        return [QuerySQLDatabaseTool(db=self.db)]

toolkit = CustomSQLDatabaseToolkit(db=db, llm=llm)

# Define the system message
system_message = SystemMessagePromptTemplate.from_template(
"""
You are an AI assistant for querying a MySQL database named {db_name}.
Your responses should be formatted for readability, using line breaks and bullet points where appropriate.
When listing items, use bulleted lists.

Database Structure Information:
[ Put in some details about the database structure here to help the LLM understand the database ]

For cohort analysis queries:
1. First segment users by their join date (usually by month or quarter)
2. Track their subsequent activities in later time periods
3. Use DATE_FORMAT() for grouping timestamps
4. Use DATEDIFF() or PERIOD_DIFF() for calculating time between events

Always strive for clarity and conciseness in your responses.
When querying for table names, use the SHOW TABLES command. To get information about a table's structure,
use the DESCRIBE command followed by the table name. When providing SQL queries, do not wrap them in code
blocks or backticks; instead, provide the raw SQL query directly.
"""
)

# Create the chat prompt
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")
])

# Create the agent and executor
agent = create_openai_functions_agent(
    llm=llm,
    prompt=chat_prompt,
    tools=toolkit.get_tools()
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
    handle_parsing_errors=True
)

def chat_interface():
    print("Welcome to the MySQL Database Query Chat Interface!")
    print("You can ask questions about your database in natural language.")
    print("Type 'exit' to quit the chat.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            logger.info(f"Processing user input: {user_input}")
            response = agent_executor.invoke({"input": user_input, "db_name": db_name})
            print("\nAgent:")
            print(response['output'])
            logger.info("Agent response provided successfully")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            print(f"An error occurred: {str(e)}")
            print("Please check the logs for more detailed information.")

if __name__ == "__main__":
    chat_interface()