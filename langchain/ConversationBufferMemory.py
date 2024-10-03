import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import OPENAI_API_KEY

# Initialize the LLM with the API key
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create the chain
chain = prompt | llm

# Create a simple message history
conversation_memory = ConversationBufferMemory()

# Create RunnableWithMessageHistory
conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda: conversation_memory.chat_memory,
    input_messages_key="input",
    verbose=True,
    history_messages_key="history"
)

# Example usage
while True:
    user_input = input(f"\n\nUser: ")
    if user_input.lower() == 'exit':
        break

    result = conversation_chain.invoke({"input": user_input})

    print(f"AI:", result.content)
    print(f"\n\n******\nChat history:\n{conversation_memory.chat_memory}\n******")