# updated with ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Local LLM via Ollama - completely free, no API key needed
llm = ChatOllama(
    model="llama3.2:3b",  # or "gemma2:2b" if you pulled that
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer based ONLY on the provided context. Be concise and accurate."),
    ("human", "Question: {query}\n\nContext:\n{context}"),
])

def generate_answer(query: str, retrieved: list):
    if not retrieved:
        return "No relevant information found in the uploaded document."
    
    context = "\n\n".join([
        f"Passage (similarity {score:.2f}):\n{chunk}"
        for chunk, score, _ in retrieved
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "context": context})
    
    return response.content.strip()