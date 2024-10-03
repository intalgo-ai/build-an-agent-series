import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from config import OPENAI_API_KEY

# Initialize the LLM with the API key
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Here's the recent conversation history:\n{buffer}"),
    ("human", "{input}")
])

# Create the chain
chain = prompt | llm

# Create a simple message history
conversation_memory = ConversationBufferWindowMemory(k=2)

# Create a simple chain that doesn't manage history
conversation_chain = RunnablePassthrough() | chain

# Example usage
while True:
    user_input = input(f"\n\nUser: ")
    if user_input.lower() == 'exit':
        break

    result = conversation_chain.invoke({
        "input": user_input,
        "buffer": conversation_memory.buffer
    })

    print(f"AI:", result.content)

    # Manually update the memory
    conversation_memory.save_context({"input": user_input}, {"output": result.content})

    print(f"\n\n******\nChat history:\n{conversation_memory.chat_memory}\n\nBuffer (message history - 2 pairs configured by setting the value of 'k'):\n{conversation_memory.buffer}\n\n******")
