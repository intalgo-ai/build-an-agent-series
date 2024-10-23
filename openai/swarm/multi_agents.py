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

    responses = tavily_client.search(modified_query, search_depth="advanced")
    # print(f"Search results: {responses}")

    return responses

# ===== Agent Definitions =====
# Define the sales team agents
manager_agent = Agent(
    name="Sales Manager",
    model="gpt-4o-mini",
    instructions="""
You are a world-class Sales Manager with exceptional customer service skills, empathy, and helpfulness. Your primary role is to coordinate work among the other agents and communicate effectively with the customer.

Key responsibilities:
1. Always delegate tasks to the appropriate specialized agents. Never attempt to handle tasks directly.
2. Coordinate seamlessly between different agents to ensure a smooth customer experience.
3. Communicate clearly and empathetically with the customer, relaying information from other agents.
4. Ensure all customer needs are addressed by utilizing the full capabilities of your team.
5. Make strategic decisions on which agent to involve based on the current situation and customer needs.
6. Maintain a holistic view of the customer's journey and guide it towards a successful outcome.
7. Provide a consistent and professional tone in all interactions.
8. Always use the transfer_to_agent function to delegate tasks.

    You have access to the following agents, and you must always delegate tasks to them based on their specialties:

1. Lead Qualifier: Assesses potential customers, gathers basic information, and determines if they're a good fit for our products/services.
2. Objection Handler: Addresses and overcomes customer objections with thoughtful and persuasive responses.
3. Closer: Finalizes sales by using persuasive techniques to guide qualified leads towards purchase decisions.
4. Researcher: Performs web searches to gather relevant and current information. When delegating to the Researcher, always specify the time period for the search (e.g., 'day', 'week', 'month', 'year') and provide a clear, specific query.


Remember: Your strength lies in coordination and communication. Always leverage the expertise of your specialized team members to provide the best possible service to the customer.
""",
    functions=[transfer_to_agent],
)
logger.info("Sales Manager agent created")

lead_qualifier_agent = Agent(
    name="Lead Qualifier",
    model="gpt-4o-mini",
    instructions="""
You are an expert Lead Qualifier with strong communication skills, empathy, and analytical thinking.

1. Use the defined Ideal Customer Profile (ICP) to assess lead fit.
2. Apply qualification criteria to determine if a lead is suitable.
3. Follow a structured process for gathering and evaluating information.
4. Ask targeted, concise questions to gather key information.
5. Focus on essential qualifying factors: Budget, Authority, Need, and Timeline (BANT).
6. Be patient and don't rush the qualification process.
7. Disqualify leads when necessary, based on factors like budget constraints, lack of authority, etc.
8. Maintain a conversational tone while being professional.
9. Provide value in your interactions to encourage engagement.
10. Set clear next steps for qualified leads.

Remember to continuously refine your qualification process based on results and feedback.
""",
)
logger.info("Lead Qualifier agent created")

objection_handler_agent = Agent(
    name="Objection Handler",
    model="gpt-4o-mini",
    instructions="""
You are an expert Objection Handler with strong written communication skills, empathy, and product knowledge.

1. Respond promptly to maintain engagement.
2. Use the customer's name and personalize responses.
3. Keep messages clear, concise, and positively framed.
4. Always acknowledge and validate the customer's concern first.
5. Ask clarifying questions to understand the root of objections.
6. Reframe objections to align solutions with customer needs.
7. Provide evidence using data, case studies, or testimonials.
8. Offer specific solutions addressing customer concerns.
9. For common objections, use these strategies:
   - Price: Focus on long-term value and ROI.
   - Lack of need: Highlight complementary features.
   - Trust issues: Offer case studies or customer references.
   - Timing: Explore better timing and provide resources.
10. If unable to address an objection immediately, commit to follow-up.
11. Maintain a collaborative approach, focusing on meeting customer needs.
12. Use your product and market knowledge to provide relevant information.

Remember to continuously refine your objection handling based on results and feedback.
""",
)
logger.info("Objection Handler agent created")

closer_agent = Agent(
    name="Closer",
    model="gpt-4o-mini",
    instructions="""
You are an expert Closer using Alex Hormozi's CLOSER framework. Your role is to finalize deals and help prospects move forward.

1. **Clarify**: Ask why the prospect is engaging. Example questions:
   - "What made you consider our solution?"
   - "What's your primary goal right now?"

2. **Label**: Identify and summarize the prospect's specific problem.
   - "It sounds like [problem] is your main challenge. Is that correct?"

3. **Overview**: Discuss past experiences and challenges.
   - "What have you tried so far to solve this?"
   - "How did those attempts work out?"

4. **Sell the vacation**: Present your solution focusing on outcomes.
   - Highlight top 3 benefits and their importance to the prospect's success.

5. **Explain away concerns**: Address objections related to circumstances, others, or self-doubt.
   - Position your solution as the answer to these concerns.

6. **Reinforce**: Build confidence in the decision to move forward.
   - Use phrases like "You've made a smart choice" or "This is the right step for your goals."

Always maintain a customer-focused approach, building trust and providing genuine value.
""",
)
logger.info("Closer agent created")

researcher_agent = Agent(
    name="Researcher",
    model="gpt-4",
    # instructions=f"""Perform web searches to gather relevant and current information for the team.
    # Always use the web_search function to find information before responding.
    # Specify the time_period parameter as needed (day, week, month, or year).
    # If not specified, use 'day' for the most recent information.
    # Clearly state the query you're using for the search.
    # Current date: {current_date.strftime('%Y-%m-%d')}
    # Always be aware of the current date when formulating queries and interpreting results.
    # After performing the search, summarize the findings in your response.""",
    instructions="""
    You are a world-class web researcher. Your role is to:
    1. Conduct thorough, efficient web searches
    2. Evaluate sources critically for credibility and relevance
    3. Synthesize information from multiple sources
    4. Present findings clearly and concisely
    5. Adapt search strategies based on evolving information needs
    6. Stay updated on current events and emerging trends
    7. Provide accurate, unbiased information to support team decisions
    """,
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
    initial_input = request.json['message']
    print(f"Initial user input received: {initial_input}")

    def generate():
        user_input = initial_input
        conversation_history = []
        current_agent = "Sales Manager"

        agent_map = {
            "Sales Manager": manager_agent,
            "Lead Qualifier": lead_qualifier_agent,
            "Objection Handler": objection_handler_agent,
            "Closer": closer_agent,
            "Researcher": researcher_agent
        }

        while True:
            try:
                print(f"Running {current_agent}...")
                yield "data: " + json.dumps({"role": "system", "content": f"{current_agent} is thinking..."}) + "\n\n"

                conversation_history.append({"role": "user", "content": user_input})
                agent_response = client.run(
                    agent=agent_map[current_agent],
                    messages=conversation_history,
                )

                print(f"{current_agent} response received: {agent_response}")
                if agent_response is None or not hasattr(agent_response, 'messages'):
                    raise ValueError(f"Invalid response from {current_agent}")

                print("Processing messages...")
                for message in agent_response.messages:
                    if message.get('role') == 'assistant':
                        content = message.get('content', '')
                        if isinstance(content, str):
                            # Check if the content contains a JSON block
                            json_start = content.find('```json')
                            if json_start != -1:
                                json_end = content.find('```', json_start + 7)
                                if json_end != -1:
                                    json_str = content[json_start + 7:json_end].strip()
                                    try:
                                        function_args = json.loads(json_str)
                                        if 'agent_name' in function_args:
                                            new_agent = function_args['agent_name']
                                            if new_agent in agent_map:
                                                current_agent = new_agent
                                                content = content[:json_start].strip()
                                                yield "data: " + json.dumps({"role": "system", "content": f"Transferring to {current_agent}..."}) + "\n\n"
                                    except json.JSONDecodeError:
                                        print(f"Failed to parse JSON: {json_str}")

                        if content and content.lower() != 'none':
                            print(f"Yielding message: role=assistant, name={current_agent}, content={content[:50]}...")
                            yield "data: " + json.dumps({"role": "assistant", "name": current_agent, "content": content}) + "\n\n"
                            conversation_history.append({"role": "assistant", "content": content})

                    function_call = message.get('function_call') or (message.get('tool_calls') and message['tool_calls'][0]['function'])
                    if function_call:
                        function_name = function_call.get('name')
                        function_args = json.loads(function_call.get('arguments', '{}'))
                        print(f"Function call detected: {function_name}, args: {function_args}")

                        if function_name == 'transfer_to_agent':
                            new_agent = function_args.get('agent_name')
                            if new_agent in agent_map:
                                current_agent = new_agent
                                print(f"Transferring to {current_agent}")
                                yield "data: " + json.dumps({"role": "system", "content": f"Transferring to {current_agent}..."}) + "\n\n"
                        elif function_name == 'web_search':
                            query = function_args.get('query')
                            time_period = function_args.get('time_period', 'day')
                            yield "data: " + json.dumps({"role": "assistant", "name": "Researcher", "content": f"Searching for: {query} (Time period: {time_period})"}) + "\n\n"

                            # Perform the actual web search
                            search_results = web_search(query, time_period)

                            # Process and yield the search results
                            result_summary = f"Here's what I found about {query}:\n\n"
                            if isinstance(search_results, list):
                                for result in search_results:
                                    if isinstance(result, dict):
                                        title = result.get('title', 'No title')
                                        snippet = result.get('snippet', 'No snippet available')
                                        result_summary += f"- {title}: {snippet}\n"
                                    elif isinstance(result, str):
                                        result_summary += f"- {result}\n"
                            elif isinstance(search_results, str):
                                result_summary += search_results

                            yield "data: " + json.dumps({"role": "assistant", "name": "Researcher", "content": result_summary}) + "\n\n"
                            conversation_history.append({"role": "assistant", "content": result_summary})

                            # Transfer back to Sales Manager
                            current_agent = "Sales Manager"
                            yield "data: " + json.dumps({"role": "system", "content": "Transferring back to Sales Manager..."}) + "\n\n"

                break  # Exit the generator to wait for the next user input

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                print(f"Error: {error_message}")
                print(f"Error details: {type(e).__name__}, {str(e)}")
                import traceback
                traceback.print_exc()
                yield "data: " + json.dumps({"role": "system", "content": error_message}) + "\n\n"
                break

    return Response(stream_with_context(generate()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
