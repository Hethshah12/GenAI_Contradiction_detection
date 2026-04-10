"""
Document Q&A chat using hybrid retrieval + Groq LLM.

Hallucination reduction strategies used:
1. Source grounding — LLM is only given retrieved chunks, not full doc
2. Citation enforcement — LLM must cite which section its answer comes from
3. Uncertainty flagging — LLM told to say "I don't know" if context is insufficient
4. Temperature 0 — deterministic, less creative/hallucinatory
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

QA_SYSTEM_PROMPT = """You are a precise document analyst answering questions about a company report.

You will be given:
- A USER QUESTION
- RELEVANT SECTIONS retrieved from the document

Rules you MUST follow:
1. Answer ONLY using information from the provided sections. Do not use outside knowledge.
2. Always cite the section label (e.g. "According to Section 3...") when making a claim.
3. If the retrieved sections do not contain enough information to answer, say:
   "The document does not provide sufficient information to answer this question."
4. If the answer is ambiguous or spread across sections, explain each part separately.
5. Be concise and factual. Do not speculate.
6. If you notice any internal contradictions relevant to the question, flag them explicitly."""


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found.")
    return Groq(api_key=api_key)


def answer_question(question: str, retrieved_chunks: list) -> dict:
    """
    Answer a user question using retrieved document chunks.

    Args:
        question: user's natural language question
        retrieved_chunks: list of chunk dicts from hybrid retrieval

    Returns:
        dict with 'answer' (str) and 'sources' (list of section labels)
    """
    client = get_client()

    # Build context from retrieved chunks
    context = ""
    sources = []
    for chunk in retrieved_chunks:
        label = chunk.get("label", "Unknown")
        context += f"\n\n[{label}]:\n{chunk['text']}"
        sources.append(label)

    user_message = (
        f"USER QUESTION: {question}\n\n"
        f"RELEVANT SECTIONS FROM THE DOCUMENT:{context}\n\n"
        f"Answer the question based strictly on the sections above."
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,   # Zero temp = minimal hallucination
            max_tokens=700
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error contacting AI: {str(e)[:100]}"

    return {
        "answer": answer,
        "sources": sources
    }


def build_chat_history_messages(history: list) -> list:
    """
    Convert Streamlit chat history to Groq message format.
    history: list of {"role": "user"/"assistant", "content": str}
    """
    messages = [{"role": "system", "content": QA_SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    return messages