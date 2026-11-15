
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Load .env from project root
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)


def get_llm():
    """
    Initialize and return Groq LLM.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        api_key=api_key
    )

    return llm
