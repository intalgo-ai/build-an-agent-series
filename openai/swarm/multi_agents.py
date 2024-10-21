from swarm import Swarm, Agent
from dotenv import load_dotenv
import os
from tavily import TavilyClient
import logging
from datetime import datetime

# Add these color codes at the beginning of the file, after the imports
BLUE = "\033[94m"
ORANGE = "\033[93m"
RESET = "\033[0m"

# Add this near the top of the file, after the imports
current_date = datetime.now()

# ===== Logging Setup =====
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ===== Environment Setup =====
# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# Initialize Swarm client with potential configuration from environment variable
client = Swarm()
logger.info("Swarm client initialized")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
logger.info("Tavily client initialized")

# ===== Helper Functions =====
def transfer_to_agent(agent_name):
    agent_map = {
        "Sales Manager": manager_agent,
        "Lead Qualifier": lead_qualifier_agent,
        "Objection Handler": objection_handler_agent,
        "Closer": closer_agent,
        "Researcher": researcher_agent
    }
    agent = agent_map.get(agent_name, None)
    if agent:
        print(f"\n[System] Transferring to {agent_name}")
    return agent

def web_search(query, time_period="day"):
    current_year = current_date.year
    time_phrase = {
        "day": f"in the last 24 hours (current date: {current_date.strftime('%Y-%m-%d')})",
        "week": f"in the last week (current date: {current_date.strftime('%Y-%m-%d')})",
        "month": f"in the last month (current date: {current_date.strftime('%Y-%m-%d')})",
        "year": f"in the year {current_year}"
    }.get(time_period, f"recently (current date: {current_date.strftime('%Y-%m-%d')})")
    
    modified_query = f"{query} {time_phrase}"
    
    print(f"\n[System] Performing web search:")
    print(f"Query: '{modified_query}'")
    print(f"Time period: {time_period}")
    
    return tavily_client.search(modified_query, search_depth="advanced", time_range=time_period)

# ===== Agent Definitions =====
# Define the sales team agents
manager_agent = Agent(
    name="Sales Manager",
    model="gpt-4o-mini",
    instructions=f"""You are the sales team manager. Oversee the sales process, delegate tasks, and ensure smooth communication.
    If you need internet information, delegate to the Researcher agent.
    When delegating to the Researcher, specify the time period for the search (e.g., 'day', 'week', 'month', 'year') and provide a clear, specific query.
    Use transfer_to_agent function to delegate tasks.
    Current date: {current_date.strftime('%Y-%m-%d')}""",
    functions=[transfer_to_agent, web_search],
)
logger.info("Sales Manager agent created")

lead_qualifier_agent = Agent(
    name="Lead Qualifier",
    model="gpt-4o-mini",
    instructions="Qualify incoming leads. Assess potential, gather basic information, and determine fit for our products/services.",
)
logger.info("Lead Qualifier agent created")

objection_handler_agent = Agent(
    name="Objection Handler",
    model="gpt-4o-mini",
    instructions="Address and overcome customer objections. Provide thoughtful and persuasive responses.",
)
logger.info("Objection Handler agent created")

closer_agent = Agent(
    name="Closer",
    model="gpt-4o-mini",
    instructions="Finalize sales. Use persuasive techniques to guide qualified leads towards purchase decisions.",
)
logger.info("Closer agent created")

researcher_agent = Agent(
    name="Researcher",
    model="gpt-4o",
    instructions=f"""Perform web searches to gather relevant and current information for the team.
    Always use the web_search function to find information before responding.
    Specify the time_period parameter as needed (day, week, month, or year).
    If not specified, use 'day' for the most recent information.
    Clearly state the query you're using for the search.
    Current date: {current_date.strftime('%Y-%m-%d')}
    Always be aware of the current date when formulating queries and interpreting results.
    After performing the search, summarize the findings in your response.""",
    functions=[web_search],
)
logger.info("Researcher agent created")

# ===== Main Interaction Loop =====
print("Welcome to the AI Sales Team! (Type 'exit' to end the conversation)")
while True:
    user_input = input(f"\n{BLUE}You:{RESET} ")
    if user_input.lower() == 'exit':
        print("Thank you for using the AI Sales Team. Goodbye!")
        break

    print("\n[System] Processing your request...")
    response = client.run(
        agent=manager_agent,
        messages=[{"role": "user", "content": user_input}],
    )

    for message in response.messages:
        if message['role'] == 'assistant':
            name = message.get('name', 'Assistant')
            content = message['content']
            if content and content.strip() != "None":
                print(f"\n{ORANGE}{name}:{RESET} {content}")
        elif message['role'] == 'function':
            if message['name'] == 'web_search':
                print(f"\n[System] Web search performed")
                # Here you might want to process and display the search results
                # For now, we'll just indicate that a search was done
            else:
                print(f"\n[System] Function '{message['name']}' called")
        elif message['role'] == 'tool':
            # Only print tool messages if they contain useful information
            content = message.get('content')
            if content and content.strip() and content.strip() != "None":
                print(f"\n[Tool] {content}")
        else:
            content = message['content']
            if content and content.strip() != "None":
                print(f"\n{message['role'].capitalize()}: {content}")

print("\n[System] Conversation ended")
