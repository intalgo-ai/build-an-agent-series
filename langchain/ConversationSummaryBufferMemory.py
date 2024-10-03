import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from config import OPENAI_API_KEY

# Initialize the LLM with the API key
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# Create a ConversationSummaryBufferMemory instance
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200,
    return_messages=True
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Create the chain
chain = (
    RunnablePassthrough.assign(
        history=lambda x: memory.load_memory_variables({})["history"]
    )
    | prompt
    | llm
    | StrOutputParser()
)

def main():
    print("Chat with the AI (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Get the AI's response
        response = chain.invoke({"input": user_input})

        # Save the interaction to memory
        memory.save_context({"input": user_input}, {"output": response})

        print(f"\n\n**********\nConversation Detail:\n{memory.chat_memory}\n")
        print(f"Conversation Summary:\n{memory.moving_summary_buffer}\n\n**********\n\n")

        print(f"AI: {response}")

if __name__ == "__main__":
    main()