import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment and initialize client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """
Use ONLY the following context to answer the question. If the answer is not found, say 'Information not found.'.

{context}

Question: {question}
"""

def generate_answer(chunks: list, question: str) -> str:
    """
    Generate a grounded answer from LLM, enforcing deterministic decoding.
    """
    context = "\n---\n".join([c.page_content for c in chunks])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    # Call chat completion with explicit decoding params
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,    # no randomness
        top_p=0.1,          # restrict to top 10% tokens
        frequency_penalty=0,
        presence_penalty=0
    )
    return resp.choices[0].message.content