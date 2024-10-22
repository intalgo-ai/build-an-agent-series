from flask import Flask, request, jsonify, render_template, stream_with_context, Response
from functools import wraps
from flask_cors import CORS
from swarm import Swarm, Agent
from dotenv import load_dotenv
import os
from tavily import TavilyClient
import logging
from datetime import datetime
import markdown
import json

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
    You have access to the following agents, and you should always delegate tasks to them based on their specialties:

    1. Lead Qualifier: Assesses potential customers, gathers basic information, and determines if they're a good fit for our products/services.
    2. Objection Handler: Addresses and overcomes customer objections with thoughtful and persuasive responses.
    3. Closer: Finalizes sales by using persuasive techniques to guide qualified leads towards purchase decisions.
    4. Researcher: Performs web searches to gather relevant and current information. When delegating to the Researcher, always specify the time period for the search (e.g., 'day', 'week', 'month', 'year') and provide a clear, specific query.

    Always use the transfer_to_agent function to delegate tasks to these agents. Choose the most appropriate agent for each task to ensure efficient and effective customer interactions.

    Current date: {current_date.strftime('%Y-%m-%d')}
    Be aware of the current date when making decisions or requesting information.""",
    functions=[transfer_to_agent],
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
    model="gpt-4",
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

# Update the Flask setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    print(f"User: {user_input}")  # Log user input

    def generate():
        try:
            print("Sales Manager is thinking...")
            yield "data: " + json.dumps({"role": "system", "content": "Sales Manager is thinking..."}) + "\n\n"
            
            manager_response = client.run(
                agent=manager_agent,
                messages=[{"role": "user", "content": user_input}],
            )
            
            for message in manager_response.messages:
                if message.get('role') == 'assistant':
                    content = message.get('content')
                    if content and content.strip() != "None":
                        content_html = markdown.markdown(content)
                        print(f"Sales Manager: {content}")
                        yield "data: " + json.dumps({"role": "assistant", "name": "Sales Manager", "content": content_html}) + "\n\n"
                
                elif message.get('role') == 'function':
                    function_name = message.get('name')
                    print(f"Sales Manager is calling function: {function_name}")
                    yield "data: " + json.dumps({"role": "system", "content": f"Sales Manager is performing a {function_name}..."}) + "\n\n"

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            yield "data: " + json.dumps({"role": "system", "content": f"An error occurred: {str(e)}"}) + "\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
