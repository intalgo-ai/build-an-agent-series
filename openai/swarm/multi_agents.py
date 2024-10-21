from swarm import Swarm, Agent
from dotenv import load_dotenv
import os
from tavily import TavilyClient

# ===== Environment Setup =====
# Load environment variables from .env file
load_dotenv()

# Initialize Swarm client with potential configuration from environment variable
client = Swarm()
print("Swarm client initialized")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ===== Helper Functions =====
def transfer_to_agent(agent_name):
    return globals()[agent_name]

# Web search function
def web_search(query):
    print(f"Performing web search for: {query}")
    return tavily_client.search(query)

# ===== Agent Definitions =====
# Define the sales team agents
manager_agent = Agent(
    name="Sales Manager",
    model="gpt-4o-mini",
    instructions="You are the sales team manager. Your role is to oversee the sales process, delegate tasks to appropriate team members, and ensure smooth communication between agents.",
    functions=[transfer_to_agent],
)

lead_qualifier_agent = Agent(
    name="Lead Qualifier",
    model="gpt-4o-mini",
    instructions="Your role is to qualify incoming leads. Assess their potential, gather basic information, and determine if they're a good fit for our products or services.",
)

objection_handler_agent = Agent(
    name="Objection Handler",
    model="gpt-4o-mini",
    instructions="You specialize in addressing and overcoming customer objections. Provide thoughtful and persuasive responses to common sales objections.",
)

closer_agent = Agent(
    name="Closer",
    model="gpt-4o-mini",
    instructions="Your job is to finalize sales. Use persuasive techniques to guide qualified leads towards making a purchase decision.",
)

researcher_agent = Agent(
    name="Researcher",
    model="gpt-4o",
    instructions="You perform web searches to gather relevant information for the team..",
    functions=[web_search],
)

# ===== Example Usage =====
# Example of running the manager agent
response = client.run(
    agent=manager_agent,
    messages=[{"role": "user", "content": "We have a new lead. How should we proceed?"}],
)

# Print the last message from the response for debugging
print(response.messages[-1]["content"])
