import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    api_key=api_key  # Explicitly pass it
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the question based only on the provided context. Be concise and accurate."),
        ("human", "{query}"),
    ]
)

def generate_answer(query: str, retrieved: list):
    if not retrieved:
        return "No relevant information found in the uploaded documents."
    
    context = "\n\n".join([f"Passage: {chunk}\n(Source similarity: {score:.2f})" for chunk, score, _ in retrieved])
    
    chain = prompt_template | llm
    response = chain.invoke({"query": query, "context": context})
    
    return response.content.strip()