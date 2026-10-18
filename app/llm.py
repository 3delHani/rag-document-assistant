from typing import List

from openai import OpenAI

from app import config

NOT_FOUND_TOKEN = "NOT_FOUND"

SYSTEM_PROMPT = f"""You are a document question-answering assistant.
Answer the user's question using ONLY the context excerpts provided.
Rules:
- Do not use outside knowledge and do not guess.
- If the context does not contain the answer, reply with exactly: {NOT_FOUND_TOKEN}
- Be concise and accurate. Reply in the same language as the question."""


def generate_answer(question: str, contexts: List[str]) -> str:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")

    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
    context_block = "\n\n".join(f"[Excerpt {i + 1}]\n{c}" for i, c in enumerate(contexts))
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()
