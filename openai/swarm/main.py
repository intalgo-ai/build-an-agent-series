from swarm import Swarm, Agent
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Swarm client with potential configuration from environment variable
client = Swarm()

def transfer_to_agent(agent_name):
    return globals()[agent_name]

# Define the sales team agents (same as in multi_agents.py)
manager_agent = Agent(
    name="Sales Manager",
    instructions="You are the sales team manager. Your role is to oversee the sales process, delegate tasks to appropriate team members, and ensure smooth communication between agents.",
    functions=[transfer_to_agent],
)

lead_qualifier_agent = Agent(
    name="Lead Qualifier",
    instructions="Your role is to qualify incoming leads. Assess their potential, gather basic information, and determine if they're a good fit for our products or services.",
)

objection_handler_agent = Agent(
    name="Objection Handler",
    instructions="You specialize in addressing and overcoming customer objections. Provide thoughtful and persuasive responses to common sales objections.",
)

closer_agent = Agent(
    name="Closer",
    instructions="Your job is to finalize sales. Use persuasive techniques to guide qualified leads towards making a purchase decision.",
)

researcher_agent = Agent(
    name="Researcher",
    instructions="You are responsible for gathering information about potential clients, market trends, and competitors. Use web search capabilities to provide up-to-date and relevant information to support the sales process.",
)

# Example of the manager delegating a task to the lead qualifier
response = client.run(
    agent=manager_agent,
    messages=[
        {"role": "user", "content": "We have a new lead named John from XYZ Corp. Can you delegate the task of qualifying this lead?"}
    ],
)

print("Manager's response:", response.messages[-1]["content"])

# Assuming the manager decides to transfer to the lead qualifier
if "transfer_to_agent" in response.messages[-1].get("function_call", {}).get("name", ""):
    lead_qualifier_response = client.run(
        agent=lead_qualifier_agent,
        messages=[
            {"role": "user", "content": "Please qualify the lead: John from XYZ Corp."}
        ],
    )
    print("Lead Qualifier's response:", lead_qualifier_response.messages[-1]["content"])
