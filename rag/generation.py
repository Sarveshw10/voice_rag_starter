# rag/generation.py - Phase 2: Groq API (llama-3.3-70b)
import os
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ── LLM setup ─────────────────────────────────────────────────────────────────
_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your .env file.\n"
        "Get a free key at https://console.groq.com"
    )

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    api_key=_api_key,
)

SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Answer the user's question using ONLY the context passages provided. "
    "If the answer is not in the context, say: "
    "'I could not find that information in the uploaded document.' "
    "Be concise and accurate. Do not make up facts."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {query}\n\nContext passages:\n{context}"),
])


def generate_answer(query: str, retrieved: list) -> str:
    """
    Generate an answer from retrieved passages using Groq.

    Args:
        query:     The user's question.
        retrieved: List of (chunk_text, score, meta) tuples from retrieval.

    Returns:
        Answer string.

    Raises:
        Exception: Propagated to main.py for clean WebSocket error handling.
    """
    if not retrieved:
        return "I could not find relevant information in the uploaded document."

    context_parts = []
    for chunk, score, meta in retrieved:
        source = meta.get("doc", "unknown")
        context_parts.append(f"[Source: {source} | Relevance: {score:.2f}]\n{chunk}")

    context = "\n\n---\n\n".join(context_parts)

    log.debug("Calling Groq with %d context passages.", len(retrieved))
    chain    = prompt | llm
    response = chain.invoke({"query": query, "context": context})
    answer   = response.content.strip()

    if not answer:
        return "The model returned an empty response. Please try again."

    return answer
