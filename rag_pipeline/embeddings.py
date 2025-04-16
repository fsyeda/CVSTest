import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model to use
EMBED_MODEL = "text-embedding-3-small"

def get_embedding(text: str) -> list[float]:
    #Generate embedding vector for a given text using OpenAI API.

    resp = client.embeddings.create(input=text, model=EMBED_MODEL)
    # The API returns a list of data entries; we take the first embedding
    return resp.data[0].embedding