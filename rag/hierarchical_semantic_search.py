from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
import os
from collections import defaultdict

# Load environment variables from the parent directory
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index = pc.Index('insert_your_index_name_here') # Ensure you put in the name of the index that you created in pinecone here

def semantic_search(question, top_k=5):
    # Generate embedding for the question
    question_embedding = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Search Pinecone index
    search_results = index.query(
        namespace="hierarchical_chunking",
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True
    )

    print(f"\nTop {top_k} relevant chunks for question: '{question}'\n")
    for i, match in enumerate(search_results['matches'], 1):
        print(f"--- Match {i} (Score: {match.score:.4f}) ---")
        print(f"Section: {match.metadata.get('Section', 'N/A')}")
        print(f"Subsection: {match.metadata.get('Subsection', 'N/A')}")
        print(f"Text: {match.metadata['text']}\n")

def create_reference_list(search_results):
    """
    Creates a hierarchical reference list from search results.
    
    Args:
        search_results: The results from the Pinecone query
        
    Returns:
        tuple: (reference_list dict, formatted_references string)
    """
    reference_list = defaultdict(set)
    
    # Build reference list
    for match in search_results['matches']:
        section = match.metadata.get('Section', 'General')
        subsection = match.metadata.get('Subsection', '')
        
        if subsection:
            reference_list[section].add(subsection)
        else:
            reference_list[section].add("General")

    # Format references as string
    formatted_references = "\n\nReferences:\n"
    for section, subsections in reference_list.items():
        formatted_references += f"Section: {section}\n"
        for subsection in sorted(subsections):
            if subsection != "General":
                formatted_references += f"  - {subsection}\n"
                
    return reference_list, formatted_references

def generate_response(question, top_k=5):
    # Generate embedding for the question
    question_embedding = client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Search Pinecone index
    search_results = index.query(
        namespace="hierarchical_chunking",
        vector=question_embedding,
        top_k=top_k,
        include_metadata=True
    )
<<<<<<< HEAD

    # Prepare context from the relevant chunks
=======
    
    # Prepare context
>>>>>>> a73eead (Added a create_reference_list method to help cite references to the vector store chunks)
    context = ""
    for match in search_results['matches']:
        section = match.metadata.get('Section', 'General')
        subsection = match.metadata.get('Subsection', '')
        text = match.metadata['text']
        context += f"\nSection: {section}"
        if subsection:
            context += f"\nSubsection: {subsection}"
        context += f"\nContent: {text}\n"
    
    # Get reference list
    _, references = create_reference_list(search_results)

    prompt = f"""Based on the following context, please provide a comprehensive answer to the question.
    If the context doesn't contain enough information, please say so.
    After your response, please include the following reference list:
    {references}

    Context:
    {context}

    Question: {question}

    Please format your response with the answer first, followed by the reference list.
    """

    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that provides detailed, accurate answers based on the given context. Always include the provided reference list at the end of your response."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    # Test question
    question = "What are the key components of a digital strategy?"

    print("\n=== Relevant Chunks ===")
    semantic_search(question)

    print("\n=== AI Generated Response ===")
    answer = generate_response(question)
    print(answer)

