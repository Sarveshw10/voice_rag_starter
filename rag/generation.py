# rag/generation.py - Phase 4: Async streaming generation
import os
import logging
from typing import AsyncGenerator
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

_api_key = os.getenv("GROQ_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GROQ_API_KEY is not set. Add it to your .env file.\n"
        "Get a free key at https://console.groq.com"
    )

# Main LLM — streaming enabled
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    api_key=_api_key,
    streaming=True,       # Phase 4: tokens flow out as they're generated
)

# Fast small model for query rewriting — no streaming needed here
rewrite_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=80,
    api_key=_api_key,
)

SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Answer the user's question using ONLY the context passages provided. "
    "If the answer is not in the context, say: "
    "'I could not find that information in the uploaded document.' "
    "Be concise and accurate. Do not make up facts. "
    "Use conversation history to understand follow-up questions."
)

REWRITE_PROMPT = (
    "You are a search query cleaner. "
    "The input is a voice-transcribed question that may contain filler words, "
    "repetitions, or speech artifacts. "
    "Rewrite it as a clean, concise search query (max 15 words). "
    "Return ONLY the rewritten query — no explanation, no punctuation at the end."
)


def rewrite_query(raw_query: str) -> str:
    """Clean a noisy STT query using the fast small model."""
    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=raw_query),
    ]
    response = rewrite_llm.invoke(messages)
    cleaned  = response.content.strip().strip(".")
    return cleaned if cleaned else raw_query


async def generate_answer_stream(
    query: str,
    retrieved: list,
    history: list | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream answer tokens one by one from Groq.

    Yields individual token strings as they arrive so the WebSocket
    can forward them to the browser in real time.

    Args:
        query:     The (rewritten) user question.
        retrieved: List of (chunk_text, score, meta) from retrieval.
        history:   Past Q&A turns for conversation memory.

    Yields:
        Token strings (single words or word-pieces).
    """
    if not retrieved:
        yield "I could not find relevant information in the uploaded document."
        return

    # Build context
    context = "\n\n---\n\n".join(
        f"[Source: {meta.get('doc','unknown')} | Relevance: {score:.2f}]\n{chunk}"
        for chunk, score, meta in retrieved
    )

    # Build message list with history
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for turn in (history or []):
        messages.append(HumanMessage(content=turn["query"]))
        messages.append(AIMessage(content=turn["answer"]))
    messages.append(HumanMessage(
        content=f"Question: {query}\n\nContext passages:\n{context}"
    ))

    log.debug("Streaming from Groq — %d passages, %d history turns.",
              len(retrieved), len(history or []))

    # astream() yields AIMessageChunk objects — we extract the text content
    async for chunk in llm.astream(messages):
        token = chunk.content
        if token:
            yield token
