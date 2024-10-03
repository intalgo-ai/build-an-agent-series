import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationSummaryMemory
from config import OPENAI_API_KEY

# Initialize the LLM with the API key
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# Create a ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

# Define a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Here's a summary of the conversation so far:\n{history}"),
    ("human", "{input}")
])

# Create the chain
chain = prompt | llm

# Create a simple chain that uses the memory
conversation_chain = RunnablePassthrough.assign(
    history=lambda x: memory.load_memory_variables({})["history"]
) | chain

# Example usage
while True:
    user_input = input(f"\n\nUser: ")
    if user_input.lower() == 'exit':
        break

    result = conversation_chain.invoke({"input": user_input})

    print(f"AI:", result.content)

    # Manually update the memory
    memory.save_context({"input": user_input}, {"output": result.content})

    # Print the current state of the memory
    print(f"\n\n******\nMemory Summary:\n{memory.load_memory_variables({})['history']}\n\n******")