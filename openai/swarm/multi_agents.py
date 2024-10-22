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

    return tavily_client.search(modified_query, search_depth="advanced")

# ===== Agent Definitions =====
# Define the sales team agents
manager_agent = Agent(
    name="Sales Manager",
    model="gpt-4o-mini",
    instructions=f"""You are the sales team manager. Your primary role is to delegate tasks to the appropriate team members.

    You have access to the following agents, and you must always delegate tasks to them based on their specialties:

    1. Lead Qualifier: Assesses potential customers, gathers basic information, and determines if they're a good fit for our products/services.
    2. Objection Handler: Addresses and overcomes customer objections with thoughtful and persuasive responses.
    3. Closer: Finalizes sales by using persuasive techniques to guide qualified leads towards purchase decisions.
    4. Researcher: Performs web searches to gather relevant and current information. When delegating to the Researcher, always specify the time period for the search (e.g., 'day', 'week', 'month', 'year') and provide a clear, specific query.

    For every user input, your response should:
    1. Briefly acknowledge the user's query or concern (1-2 sentences maximum).
    2. Use the transfer_to_agent function to delegate the task to the most appropriate agent.

    Current date: {current_date.strftime('%Y-%m-%d')}
    Be aware of the current date when deciding which agent to delegate to and what information to include in the delegation.""",
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
    is_first_message = request.json.get('isFirstMessage', False)
    print(f"User input received: {user_input}")

    def generate():
        try:
            if not is_first_message:
                print("Running manager agent...")
                yield "data: " + json.dumps({"role": "system", "content": "Sales Manager is thinking..."}) + "\n\n"

            manager_response = client.run(
                agent=manager_agent,
                messages=[{"role": "user", "content": user_input}],
            )

            print(f"Manager response received: {manager_response}")
            if manager_response is None or not hasattr(manager_response, 'messages'):
                raise ValueError("Invalid response from manager agent")

            print("Processing messages...")
            for message in manager_response.messages:
                if message.get('role') == 'assistant':
                    content = message.get('content')
                    if content:
                        print(f"Yielding message: role=assistant, content={content[:50]}...")
                        yield "data: " + json.dumps({"role": "assistant", "name": "Sales Manager", "content": content}) + "\n\n"

                function_call = message.get('function_call') or (message.get('tool_calls') and message['tool_calls'][0]['function'])
                if function_call:
                    function_name = function_call.get('name')
                    function_args = json.loads(function_call.get('arguments', '{}'))
                    print(f"Function call detected: {function_name}, args: {function_args}")

                    if function_name == 'transfer_to_agent':
                        agent_name = function_args.get('agent_name')
                        if agent_name == 'Researcher':
                            query = function_args.get('query', user_input)
                            researcher_message = f"Oh, I'll need to search for '{query}'"
                            print(f"Yielding researcher message: {researcher_message}")
                            yield "data: " + json.dumps({"role": "assistant", "name": "Researcher", "content": researcher_message}) + "\n\n"

                            print("Calling Researcher agent...")
                            researcher_response = client.run(
                                agent=researcher_agent,
                                messages=[{"role": "user", "content": query}],
                            )

                            if researcher_response and hasattr(researcher_response, 'messages'):
                                for r_message in researcher_response.messages:
                                    r_content = r_message.get('content')
                                    if r_content and isinstance(r_content, str):
                                        print(f"Yielding researcher message: {r_content[:50]}...")
                                        yield "data: " + json.dumps({"role": "assistant", "name": "Researcher", "content": r_content}) + "\n\n"
                                    elif r_content and isinstance(r_content, dict) and 'results' in r_content:
                                        # This is the search result
                                        num_results = len(r_content['results'])
                                        summary = f"I've found {num_results} relevant sources. Here's a brief summary:"
                                        for i, result in enumerate(r_content['results'][:3], 1):  # Limit to top 3 results
                                            summary += f"\n\n{i}. {result['title']}\n   {result['content'][:100]}..."
                                        print(f"Yielding researcher summary: {summary[:50]}...")
                                        yield "data: " + json.dumps({"role": "assistant", "name": "Researcher", "content": summary}) + "\n\n"
                            else:
                                print("Invalid or empty response from Researcher agent")
                                yield "data: " + json.dumps({"role": "system", "content": "Researcher couldn't find any information."}) + "\n\n"

            print("Processing complete")

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(f"Error: {error_message}")
            print(f"Error details: {type(e).__name__}, {str(e)}")
            import traceback
            traceback.print_exc()
            yield "data: " + json.dumps({"role": "system", "content": error_message}) + "\n\n"

    return Response(stream_with_context(generate()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
